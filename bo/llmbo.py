

from utils import *
import os
from jax import vmap
import copy
from jaxopt import ScipyBoundedMinimize as bounded_solver
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
import uuid
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.optimize import minimize as minimize_mo
from reccomender import * 

def llmbo(
    f,
    f_aq,
    problem_data,
):
    path = problem_data["file_name"]



    try:
        os.mkdir(path)
    except FileExistsError:
        shutil.rmtree(path)
        os.mkdir(path)
    
    data_path = path + "/res.json"

    sample_initial = problem_data["sample_initial"]
    gp_ms = problem_data["gp_ms"]

    x_bounds = f.bounds
    try:
        det_init = problem_data["deterministic_initial"]
    except:
        det_init = 'false'
    if det_init == 'true':
        lhs_key = 0 # key for deterministic initial sample for expectation over functions
        jnp_samples = lhs(jnp.array(x_bounds), sample_initial,lhs_key)
        samples = []
        for i in range(sample_initial):
            samples.append(list([float(s) for s in jnp_samples[i]]))
    elif det_init == 'false':
        samples = numpy_lhs(jnp.array(x_bounds), sample_initial)

    problem_data['deterministic_initial'] = str(det_init)

    data = {"data": []}

    for sample in samples:
        res = f(sample)
        run_info = {
            "id": str(uuid.uuid4()),
            "inputs": list(sample),
            "objective": res + np.random.normal(0,problem_data['noise'])
        }

        data["data"].append(run_info)


    save_json(data, data_path)


    data = read_json(data_path)
    for i in range(len(data['data'])):
        data['data'][i]['regret'] = (f.f_opt - jnp.max(jnp.array([data['data'][j]['objective'] for j in range(i+1)]))).item()

    # print(data)

    problem_data['f_opt'] = (f.f_opt)
    data["problem_data"] = problem_data
    alternatives = int(problem_data["alternatives"])
    save_json(data, data_path)

    iteration = len(data["data"]) - 1
    if problem_data['human_behaviour'] == 'llmbo' and problem_data['llm_location'] != 'remote':

        llm = Llama(model_path=problem_data['llm_location'],n_ctx=4096,n_gpu_layers=-1,n_threads=4)
    while len(data['data']) < problem_data['max_iterations']:
        
            
        start_time = time.time()
        data = read_json(data_path)
        inputs, outputs, cost = format_data(data)
        mean_outputs = np.mean(outputs)
        std_outputs = np.std(outputs)
        if std_outputs != 0:
            outputs = (outputs - mean_outputs) / std_outputs

        mean_inputs = np.mean(inputs, axis=0)
        std_inputs = np.std(inputs, axis=0)
        inputs = (inputs - mean_inputs) / std_inputs

        bounds = []
        for i in range(len(x_bounds)):
            lb = float((x_bounds[i][0] - mean_inputs[i]) / std_inputs[i])
            ub = float((x_bounds[i][1] - mean_inputs[i]) / std_inputs[i])
            bounds.append([lb,ub])

        d = len(inputs[0])
        f_best = np.max(outputs)
        gp = build_gp_dict(*train_gp(inputs, outputs, gp_ms,its=10000,noise=problem_data['noisy']))


            
        util_args = (gp, f_best)
        if problem_data['noisy'] == True:
            n_gps = problem_data['letham_gps']
            gp_list = []
            f_best_list = []
            key = jax.random.PRNGKey(0)
            for i in range(n_gps):
                mean,std = inference(gp,inputs)
                det_outputs = []
                for i in range(len(mean)):
                    normal = tfd.Normal(loc=mean[i],scale=std[i])
                    det_outputs.append(normal.sample(seed=key))
                    key,subkey = jax.random.split(key)
                det_outputs = np.array([det_outputs]).T
                new_gp = build_gp_dict(*train_gp(inputs,det_outputs,int(gp_ms/2),noise=False))
                gp_list.append(new_gp) 
                f_best_list.append(np.max(det_outputs))

            util_args = (gp_list, f_best_list)


        aq = vmap(f_aq, in_axes=(0, None))


        if problem_data['plot'] == True:
            fig,ax = plt.subplots(2,1,figsize=(10,6),constrained_layout=True,sharex=True)
            x_plot = np.linspace(x_bounds[0][0],x_bounds[0][1],200)
            y_plot = f.eval_vector(x_plot)
            ax[0].plot(x_plot,y_plot,c='k',label='True Function',ls='dashed')
            for dat in data['data']:
                ax[0].scatter(dat['inputs'],dat['objective'],c='k',marker='x')

            if problem_data['acquisition_function'] == 'LETHAM':
                for gp in gp_list:
                    x_plot_gp = np.linspace(bounds[0][0],bounds[0][1],200)
                    gp_m,gp_s = inference(gp,x_plot_gp)
                    gp_m = gp_m * std_outputs + mean_outputs
                    gp_s = gp_s * std_outputs
                    ax[0].plot(x_plot,gp_m,c='k')
                    ax[0].fill_between(x_plot,gp_m - 2*gp_s,gp_m + 2*gp_s,color='k',lw=0,alpha=0.2)

                v_EI = vmap(logEI, in_axes=(0, None))
                for gp in gp_list:
                    aq_plot = v_EI(x_plot_gp,(gp,f_best))
                    ax[1].plot(x_plot[:len(aq_plot)],-np.array(aq_plot),c='r',alpha=0.5)

                aq_plot = aq(x_plot_gp,util_args)
                ax[1].plot(x_plot[:len(aq_plot)],-np.array(aq_plot),c='k')
                ax[1].set_yscale('symlog')


            else:
                x_plot_gp = np.linspace(bounds[0][0],bounds[0][1],200)
                gp_m,gp_s = inference(gp,x_plot_gp)
                gp_m = gp_m * std_outputs + mean_outputs
                gp_s = gp_s * std_outputs
                ax[0].plot(x_plot,gp_m,c='k')
                ax[0].fill_between(x_plot,gp_m - 2*gp_s,gp_m + 2*gp_s,color='k',lw=0,alpha=0.2)
                aq_plot = aq(x_plot_gp,util_args)
                ax[1].plot(x_plot[:len(aq_plot)],-np.array(aq_plot),c='k',alpha=0.5)



            ax[0].legend(frameon=False)


            # fig.savefig('true_function.png',dpi=200)
            fig.savefig(path + '/plot_' + str(iteration) + '.png',dpi=200)

        # optimising the aquisition of inputs, disregarding fidelity
        print("Optimising utility function...")
        upper_bounds_single = jnp.array([b[1] for b in bounds])
        lower_bounds_single = jnp.array([b[0] for b in bounds])

        opt_bounds = (lower_bounds_single, upper_bounds_single)
        s_init = jnp.array(sample_bounds(x_bounds, 256))
        
        solver = bounded_solver(
            method="l-bfgs-b",
            jit=True,
            fun=f_aq,
            tol=1e-10,
        )

        def optimise_aq(s):
            res = solver.run(init_params=s, bounds=opt_bounds, args=util_args)
            aq_val = res.state.fun_val
            print('Iterating utility took: ', res.state.iter_num, ' iterations with value of ',aq_val)
            x = res.params
            return aq_val, x

        aq_vals = []
        xs = []
        for s in s_init:
            aq_val, x = optimise_aq(s)
            aq_vals.append(aq_val)
            xs.append(x)

        x_opt_aq = xs[jnp.argmin(jnp.array(aq_vals))]

        if problem_data['human_behaviour'] == 'trusting':
            x_opt = x_opt_aq
            x_opt = list((np.array(x_opt) * std_inputs) + mean_inputs)

        else:

            n_opt = int(len(bounds) * (alternatives-1))
            upper_bounds = jnp.repeat(upper_bounds_single, alternatives-1)
            lower_bounds = jnp.repeat(lower_bounds_single, alternatives-1)
            termination = DefaultMultiObjectiveTermination(
            xtol=problem_data['NSGA_xtol'],
            ftol=problem_data['NSGA_ftol'],
            period=30,
            n_max_gen=10000,
            n_max_evals=100000)

            algorithm = NSGA2(
                pop_size=50,
                n_offsprings=10,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True,
            )


            class MO_aq(Problem):
                def __init__(self):
                    super().__init__(
                        n_var=n_opt,
                        n_obj=2,
                        n_ieq_constr=0,
                        xl=np.array(lower_bounds),
                        xu=np.array(upper_bounds),
                    )

                def _evaluate(self, x, out, *args, **kwargs):
                    x_sols = jnp.array(jnp.split(x, alternatives-1, axis=1))
                    d = x_sols.shape[0]
                    aq_list = np.sum([aq(x_i, util_args) for x_i in x_sols], axis=0)
                    if d == 1:
                        app = jnp.array([[x_opt_aq for i in range(len(x_sols[0,:,0]))]]).T
                    else:
                        app = jnp.array([[x_opt_aq for i in range(len(x_sols[0,:,0]))]])

                    x_sols = jnp.append(x_sols, app, axis=0)

                    K_list = []
                    for i in range(len(x_sols[0])):
                        set = jnp.array([x_sols[j][i] for j in range(alternatives)])
                        K = gp["posterior"].prior.kernel.gram(set).matrix
                        K = jnp.linalg.det(K)
                        K_list.append(K)
                    K_list = np.array(K_list)

                    out["F"] = [aq_list, -K_list]

            problem = MO_aq()
            res = minimize_mo(
                problem, algorithm, termination, seed=1, save_history=True, verbose=True
            )

        
            F = res.F
            X = res.X

            AQ = F[:, 0]
            D = F[:, 1]

            aq_norm = (AQ - np.min(AQ)) / (np.max(AQ) - np.min(AQ))
            d_norm = (D - np.min(D)) / (np.max(D) - np.min(D))
            # utopia_index
            distances = np.sqrt(aq_norm**2 + d_norm**2)

            x_best_aq = jnp.append(X[np.argmin(AQ)], x_opt_aq)
            x_best_d = jnp.append(X[np.argmin(D)], x_opt_aq)
            x_best_utopia = jnp.append(X[np.argmin(distances)], x_opt_aq)

            best_aq_sol = np.argmin(AQ)
            best_D_sol = np.argmin(D)
            utopia_sol = np.argmin(distances)

            x_alternates = list(jnp.split(x_best_utopia, alternatives))
            x_alternates = [x_alternates[i].tolist() for i in range(alternatives)]
            for i in range(alternatives):
                # unnormalise alternative solutions
                x_alternates[i] = list((np.array(x_alternates[i]) * std_inputs) + mean_inputs)

            if problem_data['human_behaviour'] == 'llmbo':
                x_names = problem_data['x_names']
                expertise = problem_data['expertise']
                obj_desc = problem_data['objective_description']
                aq_list = np.array([float(-f_aq(jnp.array(x_alternates[i]), util_args)) for i in range(alternatives)])
                aq_list += np.abs(np.min(aq_list))
                aq_list = aq_list.tolist()
                aq_list = [np.round(aq_list[i],3) for i in range(alternatives)]
                previous_iterations = 5

                prompt_data = {'previous_iterations':data['data'][-previous_iterations:]}

                prev_just = problem_data['include_previous_justification']

                prompt = create_prompt(f,x_names,x_alternates,aq_list,prompt_data,expertise,obj_desc,prev_just)

                print(prompt)
                if problem_data['llm_location'] == 'remote':
                    llm = 'gpt-3.5-turbo-0613'

                response = run_prompt(llm,prompt)
                print(response)

                try:
                    x_opt = np.array([x_alternates[response['choice']-1]])
                    bad_flag = False
                except:
                    x_opt = x_opt_aq
                    x_opt = list((np.array(x_opt) * std_inputs) + mean_inputs)
                    bad_flag = True

            if problem_data['human_behaviour'] == 'expert':
                f_utopia = []
                for i in range(alternatives):
                    f_utopia.append(f(x_alternates[i]))
                x_opt = np.array([x_alternates[np.argmax(f_utopia)]])

            if problem_data['human_behaviour'] == 'adversarial':
                f_utopia = []
                for i in range(alternatives):
                    f_utopia.append(f(x_alternates[i]))
                x_opt = np.array([x_alternates[np.argmin(f_utopia)]])
            

            if problem_data['human_behaviour'].__class__ == float:
                if problem_data['human_behaviour'] < 0 or problem_data['human_behaviour'] > 1:
                    raise ValueError("Human behaviour must be between 0 and 1")
                
                f_utopia = []
                for i in range(alternatives):
                    f_utopia.append(f(x_alternates[i]))

                best_index = np.argmax(f_utopia)
                probability_of_correct = np.random.uniform()
                if probability_of_correct < problem_data['human_behaviour']:
                    x_opt = np.array([x_alternates[best_index]])
                else:
                    x_tests = np.delete(x_alternates,best_index,axis=0)
                    x_opt = np.array([x_alternates[np.random.randint(0,alternatives-1)]])

        if d == 1:
            x_opt = [x_opt[0].item()]

        iteration += 1

        if d > 1 and problem_data['human_behaviour'] != 'trusting' and problem_data['human_behaviour'] != 'llmbo':
            x_opt = x_opt[0]
        
        elif d > 1 and problem_data['human_behaviour'] == 'trusting':
            x_opt = [float(x) for  x in x_opt]

        elif d > 1 and problem_data['human_behaviour'] == 'llmbo' and bad_flag == True:
            x_opt = [float(x) for  x in x_opt]

        elif d > 1 and problem_data['human_behaviour'] == 'llmbo' and bad_flag == False:
            x_opt = x_opt[0]

            
        print("Optimal Solution: ", x_opt)
        x_opt = [float(x_opt[i]) for i in range(len(x_opt))]

        f_eval =  f(x_opt)
        run_info = {
            "inputs": list(x_opt),
        }

        run_info["objective"] = f_eval + np.random.normal(0,problem_data['noise'])
        run_info["id"] = str(uuid.uuid4())

        if problem_data['human_behaviour'] == 'llmbo':
            if bad_flag == False:
                run_info['reason'] = response

        
        regret = min((f.f_opt - f_eval),data['data'][-1]['regret'])
        if regret.__class__ != float:
            regret = regret.item()
        run_info["regret"] = regret

        data["data"].append(run_info)


        save_json(data, data_path)

