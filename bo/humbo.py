import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.ce_functions import *
import datetime
from utils import *
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
from tabulate import tabulate

def humbo(
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
    try:
        os.mkdir(path + '/result_plots')
    except FileExistsError:
        shutil.rmtree(path)
        os.mkdir(path + '/result_plots')


    # create csv for initial data with header of f.x_names
    pd.DataFrame(columns=f.x_names).to_excel(path+'/'+f.name+"_initial_data.xlsx",index=False)

    print('The bounds of the problem are as follows: ')

    x_bounds = f.bounds
    data_path = path + "/res.json"
    sample_initial = problem_data["sample_initial"]
    gp_ms = problem_data["gp_ms"]

    print(tabulate(headers=f.x_names,tabular_data=np.array(x_bounds).T,tablefmt='fancy_grid'))

    gp_ms = problem_data["gp_ms"]

    print('Edit the '+path+'/'+f.name+"_initial_data.xlsx"+' file with any initial solutions, ensure the solutions are within the stated bounds, then press any key.')
    input()

    # read in initial data
    initial_data = pd.read_excel(path+'/'+f.name+"_initial_data.xlsx")
    initial_data = initial_data.to_numpy()
    if len(initial_data) == 0:
        lhs_key = 0 # key for deterministic initial sample for expectation over functions
        # jnp_samples = lhs(jnp.array(x_bounds), sample_initial,lhs_key)
        jnp_samples = np.random.uniform(low=0,high=1,size=(sample_initial,len(x_bounds))) * (np.array(x_bounds)[:,1] - np.array(x_bounds)[:,0]) + np.array(x_bounds)[:,0]
        samples = []
        for i in range(sample_initial):
            samples.append(list([float(s) for s in jnp_samples[i]]))
    else:
        samples = distribute_solutions(initial_data,x_bounds,problem_data['sample_initial']-len(initial_data))
        samples = list(samples) 

    data = {"data": []}

    for sample in samples:
        res = f(sample)

        run_info = {
            "id": str(uuid.uuid4()),
            "inputs": list(sample),
            "objective": res + np.random.normal(0,problem_data['noise'])
        }

        try:
            f.plot_result(sample,path + '/result_plots/' +run_info['id']+'.png',f)
        except:
            print('Plotting of result not implemented for this function...')

        data["data"].append(run_info)

    save_json(data, data_path)


    data = read_json(data_path)
    for i in range(len(data['data'])):
        data['data'][i]['regret'] = (f.f_opt - jnp.max(jnp.array([data['data'][j]['objective'] for j in range(i+1)]))).item()

    problem_data['f_opt'] = (f.f_opt)
    data["problem_data"] = problem_data
    alternatives = int(problem_data["alternatives"])
    save_json(data, data_path)

    iteration = len(data["data"]) - 1


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

        print("Optimising utility function...")
        upper_bounds_single = jnp.array([b[1] for b in bounds])
        lower_bounds_single = jnp.array([b[0] for b in bounds])

        opt_bounds = (lower_bounds_single, upper_bounds_single)
        s_init = jnp.array(sample_bounds(bounds, 256))
        
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

        n_opt = int(len(bounds) * (alternatives-1))
        # mo_upper_bounds = jnp.repeat(upper_bounds_single, alternatives-1)
        # mo_lower_bounds = jnp.repeat(lower_bounds_single, alternatives-1)

        mo_upper_bounds = jnp.array([b[1] for b in bounds] * (alternatives-1))
        mo_lower_bounds = jnp.array([b[0] for b in bounds] * (alternatives-1))
        print(lower_bounds_single,upper_bounds_single,'\n')
        print(mo_lower_bounds,mo_upper_bounds)

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
                    xl=np.array(mo_lower_bounds),
                    xu=np.array(mo_upper_bounds),
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

        x_best_utopia = jnp.append(X[np.argmin(distances)], x_opt_aq)

        x_alternates = list(jnp.split(x_best_utopia, alternatives))
        x_alternates = [x_alternates[i].tolist() for i in range(alternatives)]


        # evaluate posterior at alternate solutions
        mean_alt = []; var_alt = []
        for i in range(len(x_alternates)):
            mean, var = inference(gp, np.array([x_alternates[i]]))
            mean_alt.append(mean[0])
            var_alt.append(var[0])

        mean_alt = np.array(mean_alt)
        var_alt = np.array(var_alt)

        mean_alt = (np.array(mean_alt) * std_outputs) + mean_outputs
        var_alt = (np.array(var_alt) * std_outputs) + mean_outputs

        for i in range(alternatives):
            x_alternates[i] = list((np.array(x_alternates[i]) * std_inputs) + mean_inputs)

        x_names = problem_data['x_names']
        aq_list = np.array([float(-f_aq(jnp.array(x_alternates[i]), util_args)) for i in range(alternatives)])
        aq_list += np.abs(np.min(aq_list))
        aq_list = aq_list.tolist()
        aq_list = [np.round(aq_list[i],3) for i in range(alternatives)]
        previous_iterations = 5

        prompt_data = {'previous_iterations':data['data'][-previous_iterations:]}
        
        try:
            os.mkdir(path + '/choices')
        except FileExistsError:
            shutil.rmtree(path + '/choices')
            os.mkdir(path + '/choices')

        for i in range(len(x_alternates)):
            f.plot_solution(x_alternates[i],path + '/choices/choice_'+str(i+1)+'.png',f)

        create_human_prompt(f,x_names,x_alternates,aq_list,prompt_data,mean_alt,var_alt)

        print('Remember to check solution visualisations in '+path+'/choices/ before making a choice.')

        choice = input('Enter choice: ')
        choice = int(choice)
        x_opt = np.array([x_alternates[choice-1]])[0]

        iteration += 1
            
        print("Optimal Solution: ", x_opt)
        x_opt = [float(x_opt[i]) for i in range(len(x_opt))]

        f_eval =  f(x_opt)
        run_info = {
            "inputs": list(x_opt),
        }

        run_info["objective"] = f_eval + np.random.normal(0,problem_data['noise'])
        run_info["id"] = str(uuid.uuid4())
        try:
            f.plot_result(sample,path + '/result_plots/' +run_info['id']+'.png',f)
        except:
            print('Plotting of result not implemented for this function...')

        
        regret = min((f.f_opt - f_eval),data['data'][-1]['regret'])
        if regret.__class__ != float:
            regret = regret.item()
        run_info["regret"] = regret

        data["data"].append(run_info)


        save_json(data, data_path)


name = input('Enter name and press Enter: ').lower()

res_path = 'bo/'+name + '/'

try:
    os.mkdir(res_path)
except FileExistsError:
    pass


# def create_SelfOpt():
#     f = SelfOpt(4)
#     return f



problem_data = {}
problem_data["sample_initial"] = 12
problem_data["gp_ms"] = 8
problem_data["alternatives"] = 4
problem_data["NSGA_xtol"] = 1e-6
problem_data["NSGA_ftol"] = 0.005
problem_data['max_iterations'] = 48

#f = Reactor(1)
f = BioProcess_Profile(4)

problem_data['x_names'] = f.x_names
problem_data['UTC'] = str(datetime.datetime.now())
problem_data['expertise'] = f.expertise
problem_data['objective_description'] = f.objective_description
problem_data['function'] = f.name
problem_data['dim'] = f.dim

aq = 'LETHAM_UCB'
problem_data["noisy"] = True
noise_std = 0.025
problem_data['noise'] = noise_std * f.y_range
problem_data['max_iterations'] = problem_data['max_iterations'] * 2
problem_data['letham_gps'] = 8

problem_data['acquisition_function'] = aq
problem_data['time_created'] = str(datetime.datetime.now())

#problem_data['llm_location'] = 'remote'
# problem_data['llm_location'] = "llama.cpp/models/13B/ggml-model-q8_0.gguf"
# problem_data['llm_location'] = "llama.cpp/models/zephyr-7b-alpha.Q4_K_M.gguf"

problem_data['include_previous_justification'] = False
file = f.name 
path = res_path + file + "/"
problem_data['file_name'] = path

aqs = {'LETHAM_UCB':LETHAM_UCB,'UCB':UCB,'EI':EI}   

humbo(
    f,
    aqs[aq],
    problem_data
)
