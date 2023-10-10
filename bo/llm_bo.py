

from utils import *
import os
from jax import vmap
from jaxopt import ScipyBoundedMinimize as bounded_solver
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
import uuid
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
    samples = numpy_lhs(jnp.array(x_bounds), sample_initial)

    data = {"data": []}

    for sample in samples:
        
        
        res = f(sample)
        run_info = {
            "id": str(uuid.uuid4()),
            "inputs": list(sample),
            "objective": res
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
    while len(data['data']) < problem_data['max_iterations']:
        
            
        start_time = time.time()
        data = read_json(data_path)
        inputs, outputs, cost = format_data(data)
        d = len(inputs[0])
        f_best = np.max(outputs)
        gp = build_gp_dict(*train_gp(inputs, outputs, gp_ms))
        util_args = (gp, f_best)


        aq = vmap(f_aq, in_axes=(0, None))

        # optimising the aquisition of inputs, disregarding fidelity
        print("Optimising utility function...")
        upper_bounds_single = jnp.array([b[1] for b in x_bounds])
        lower_bounds_single = jnp.array([b[0] for b in x_bounds])

        opt_bounds = (lower_bounds_single, upper_bounds_single)
        s_init = jnp.array(sample_bounds(x_bounds, 36))
        
        solver = bounded_solver(
            method="l-bfgs-b",
            jit=True,
            fun=f_aq,
            tol=1e-12,
            maxiter=500
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
            print(x_opt)

        else:

            n_opt = int(len(x_bounds) * (alternatives-1))
            upper_bounds = jnp.repeat(upper_bounds_single, alternatives-1)
            lower_bounds = jnp.repeat(lower_bounds_single, alternatives-1)
            termination = get_termination("n_gen", problem_data["NSGA_iters"])

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


            if problem_data['human_behaviour'] == 'llmbo':
                x_alternates = list(jnp.split(x_best_utopia, alternatives))
                x_alternates = [x_alternates[i].tolist() for i in range(alternatives)]

                # unnormalise x_alternatives for LLM 
                # bounds are 0-1 but f.var_bounds contains real
                for i in range(alternatives):
                    for j in range(len(x_alternates[i])):
                        x_alternates[i][j] = x_alternates[i][j] * (f.var_bounds[j][1] - f.var_bounds[j][0]) + f.var_bounds[j][0]


                x_names = problem_data['x_names']
                expertise = problem_data['expertise']
                obj_desc = problem_data['objective_description']
                aq_list = np.array([float(-f_aq(jnp.array(x_alternates[i]), util_args)) for i in range(alternatives)])
                aq_list += np.abs(np.min(aq_list))
                aq_list = aq_list.tolist()
                aq_list = [np.round(aq_list[i],3) for i in range(alternatives)]
                previous_iterations = 5

                temperature = 0.5
                if problem_data['gpt'] == 3.5:
                    model = 'gpt-3.5-turbo-0613'
                elif problem_data['gpt'] == 4:
                    model = 'gpt-4-0613'
                # data is the last previous iterations 
                prompt_data = {'previous_iterations':data['data'][-previous_iterations:]}
                prev_just = problem_data['include_previous_justification']
                response = json.loads(expert_reccomendation(f,x_names,x_alternates,aq_list,prompt_data,expertise,obj_desc,model,temperature,prev_just))
                # if choice is not a float then choose the best aquisition
                try:
                    if response['choice'].__class__ != float:
                        x_opt = np.array([x_alternates[-1]])
                    else:
                        x_opt = np.array([x_alternates[response['choice']-1]])
                except:
                        x_opt = np.array([x_alternates[-1]])
            
            if problem_data['human_behaviour'] == 'expert':
                f_utopia = []
                x_tests = jnp.array(jnp.split(x_best_utopia, alternatives))
                for i in range(alternatives):
                    f_utopia.append(f(x_tests[i]))
                x_opt = np.array([x_tests[np.argmax(f_utopia)]])

            if problem_data['human_behaviour'] == 'adversarial':
                f_utopia = []
                x_tests = jnp.array(jnp.split(x_best_utopia, alternatives))
                for i in range(alternatives):
                    f_utopia.append(f(x_tests[i]))
                x_opt = np.array([x_tests[np.argmin(f_utopia)]])
            

            if problem_data['human_behaviour'].__class__ == float:
                if problem_data['human_behaviour'] < 0 or problem_data['human_behaviour'] > 1:
                    raise ValueError("Human behaviour must be between 0 and 1")
                
                f_utopia = []
                x_tests = jnp.array(jnp.split(x_best_utopia, alternatives))
                for i in range(alternatives):
                    f_utopia.append(f(x_tests[i]))

                best_index = np.argmax(f_utopia)
                probability_of_correct = np.random.uniform()
                if probability_of_correct < problem_data['human_behaviour']:
                    x_opt = np.array([x_tests[best_index]])
                else:
                    x_tests = np.delete(x_tests,best_index,axis=0)
                    x_opt = np.array([x_tests[np.random.randint(0,alternatives-1)]])

        if d == 1:
            x_opt = [x_opt[0].item()]

        iteration += 1

        mu_opt,var_opt = inference(gp, jnp.array([x_opt]))


        if d > 1 and problem_data['human_behaviour'] != 'trusting':
            x_opt = x_opt[0]

        if d > 1 and problem_data['human_behaviour'] == 'trusting':

            x_opt = [float(x) for  x in x_opt]
        print(x_opt)
        
        

        print("Optimal Solution: ", x_opt)


        f_eval =  f(x_opt)
        run_info = {
            "inputs": list(x_opt),
        }

        run_info["objective"] = f_eval
        run_info["id"] = str(uuid.uuid4())
        run_info["pred_mu"] = np.float64(mu_opt)
        run_info["pred_sigma"] = np.float64(np.sqrt(var_opt))

        if problem_data['human_behaviour'] == 'llmbo':
            run_info['reason'] = response

        
        regret = (f.f_opt - max(f_eval,jnp.max(outputs)))
        if regret.__class__ != float:
            regret = regret.item()
        run_info["regret"] = regret

        data["data"].append(run_info)


        save_json(data, data_path)

