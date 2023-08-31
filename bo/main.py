from utils import *
import os
import jax.random as random
from jax import vmap 
from jaxopt import ScipyBoundedMinimize as bounded_solver
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination

from pymoo.optimize import minimize




def bo(
    f,
    f_aq,
    alternatives,
    data_path,
    problem_data,
    path,
    eval_error=True,
    printing=False,
    key=random.PRNGKey(0),
):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    err_tol = problem_data["err_tol"]
    sample_initial = problem_data["sample_initial"]
    gp_ms = problem_data["gp_ms"]
    ms_num = problem_data["ms_num"]

    x_bounds = f.bounds
    samples = numpy_lhs(jnp.array(list(x_bounds.values())), sample_initial)

    data = {"data": []}

    for sample in samples:
        sample_dict = sample_to_dict(list(sample), x_bounds)
        s_eval = sample_dict.copy()
        res = f(s_eval)
        run_info = {
            "id": res["id"],
            "inputs": sample_dict,
            "cost": jnp.float64(res["cost"]).item(),
            "objective": jnp.float64(res["objective"]).item(),
        }
        data["data"].append(run_info)

        save_json(data, data_path)

    data = read_json(data_path)
    data["problem_data"] = problem_data
    alternatives = problem_data['alternatives']

    save_json(data, data_path)

    iteration = len(data["data"]) - 1

    error = 1e10
    while error > err_tol:
        start_time = time.time()
        data = read_json(data_path)
        inputs, outputs, cost = format_data(data)
        gp = build_gp_dict(*train_gp(inputs, outputs, gp_ms))
        # x_opt_samples,f_opt_samples = global_optimum_distributions(x_bounds, gp, 10)


        n_test = 1000
        x_test = jnp.linspace(x_bounds["x"][0], x_bounds["x"][1], n_test).reshape(-1, 1)

        y_true = f.eval_vector(x_test[:,0])

        aq = vmap(f_aq, in_axes=(0, None))
        aq_vals_list = aq(x_test, (gp))

        posterior = gp['posterior']
        D = gp['D']
        latent_dist = posterior.predict(x_test, train_data=D)
        predictive_dist = posterior.likelihood(latent_dist)
        mean = predictive_dist.mean()
        cov = jnp.sqrt(predictive_dist.variance())

        error = jnp.mean((mean - y_true) ** 2)

        # optimising the aquisition of inputs, disregarding fidelity
        print("Optimising utility function...")

        # sample and normalise initial guesses
        s_init = jnp.array(sample_bounds(x_bounds, ms_num))

        args = (gp)

        # upper_bounds = jnp.array([b[1] for b in list(x_bounds.values())])
        # lower_bounds = jnp.array([b[0] for b in list(x_bounds.values())])
        # opt_bounds = (lower_bounds, upper_bounds)
        
        # solver = bounded_solver(
        #     method="l-bfgs-b",
        #     jit=True,
        #     fun=f_aq,
        #     tol=1e-12,
        #     maxiter=500,
        # )

        # def optimise_aq(s):
        #     res = solver.run(init_params=s, bounds=opt_bounds, args=args)
        #     aq_val = res.state.fun_val
        #     print('Iterating utility took: ', res.state.iter_num, ' iterations')
        #     x = res.params
        #     return aq_val, x

        # aq_vals = []
        # xs = []
        # for s in s_init:
        #     aq_val, x = optimise_aq(s)
        #     aq_vals.append(aq_val)
        #     xs.append(x)


        args = (alternatives,gp)
        n_opt = int(len(x_bounds.values()) * alternatives)
        upper_bounds = jnp.array([b[1] for b in list(x_bounds.values())])
        lower_bounds = jnp.array([b[0] for b in list(x_bounds.values())])
        upper_bounds = jnp.repeat(upper_bounds, alternatives)
        lower_bounds = jnp.repeat(lower_bounds, alternatives)

        termination = get_termination("n_gen", 100)
        
        algorithm = NSGA2(
            pop_size=30,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        class MO_aq(Problem):

            def __init__(self):
                
                super().__init__(n_var=n_opt,
                            n_obj=2,
                            n_ieq_constr=0,
                            xl=np.array(lower_bounds),
                            xu=np.array(upper_bounds))

            def _evaluate(self, x, out, *args, **kwargs):

                x_sols = jnp.split(x, alternatives,axis=1)
                K_list = []
                for i in range(len(x_sols[0])):
                    set = jnp.array([x_sols[j][i] for j in range(alternatives)])
                    K = gp['posterior'].prior.kernel.gram(set).matrix
                    K = jnp.linalg.det(K)
                    K_list.append(K)
                K_list = np.array(K_list)

                aq_list = np.sum([aq(x_i,(gp)) for x_i in x_sols],axis=0)
                out["F"] = [aq_list,-K_list]

        problem = MO_aq()
        res = minimize(problem,
                    algorithm,
                    termination,
                    seed=1,
                    save_history=True,
                    verbose=True)

        F = res.F
        X = res.X

        AQ = F[:,0]
        D = F[:,1]

        aq_norm = (AQ - np.min(AQ)) / (np.max(AQ) - np.min(AQ))
        d_norm = (D - np.min(D)) / (np.max(D) - np.min(D))
        # utopia_index
        distances = np.sqrt(aq_norm**2 + d_norm**2)



        x_best_aq = X[np.argmin(AQ)]
        x_best_d = X[np.argmin(D)]
        x_best_utopia = X[np.argmin(distances)]




        plt.figure(figsize=(7, 5))
        plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
        plt.scatter(F[np.argmin(AQ), 0], F[np.argmin(AQ), 1], s=30, c='red')
        plt.scatter(F[np.argmin(D), 0], F[np.argmin(D), 1], s=30, c='green')
        plt.scatter(F[np.argmin(distances), 0], F[np.argmin(distances), 1], s=30, c='blue')
        plt.xlabel('Sum of Acquisition Function Values')
        plt.ylabel('Solution Spread')
        plt.title("Objective Space")
        plt.savefig(path +'/'+ str(iteration+1)+"_pareto.png", dpi=400)


        random_choice = np.random.randint(0, alternatives) # should be made by a human
        x_opt = np.array([x_best_utopia[random_choice]])

        #x_opt = xs[jnp.argmin(jnp.array(aq_vals))]

        if printing == True:
            fig, axs = plt.subplots(2, 1, figsize=(8, 4))
            ax = axs[0]
            max_f = np.argmax(y_true)
            ax.plot(x_test[:,0], y_true, c="k", lw=2, label="Function", alpha=0.5)
            ax.scatter(x_test[max_f], y_true[max_f], c="k", s=40, marker='+',label='Global Optimum')
            ax.scatter(inputs, outputs, c="k", s=20, lw=0, label="Data")
            # remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("$x$")
            ax.set_ylabel("$f(x)$")
            ax.plot(x_test[:,0], mean, c="k", ls="--", lw=2, label="GP Posterior")
            ax.fill_between(
                x_test[:,0],
                mean + 2 * cov,
                mean - 2 * cov,
                alpha=0.05,
                color="k",
                lw=0,
                label="95% Confidence",
            )
            # place legend below plot
            ax.legend(
                frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5
            )

            ax = axs[1]
            aq_vals_list = -aq_vals_list
            ax.plot(x_test[:,0], aq_vals_list, c="k", lw=2, label="Acquisition Function")
            ax.fill_between(x_test[:,0], min(aq_vals_list), aq_vals_list, alpha=0.05, color="k", lw=0)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$\mathcal{U}(x)$")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.scatter(x_opt, -f_aq(x_opt,(gp)), c="k", s=20, lw=0, label="Optimum")
            for i in range(alternatives):
                ax.scatter(x_best_aq[i], -f_aq(x_best_aq[i],(gp)), c='red', s=40)
                ax.scatter(x_best_d[i], -f_aq(x_best_d[i],(gp)), c='green', s=40)
                ax.scatter(x_best_utopia[i], -f_aq(x_best_utopia[i],(gp)), c='blue', s=40)


            ax.legend(
                frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2
            ) 
            fig.tight_layout()
            plt.savefig(path + "/" + str(iteration + 1) + ".png", dpi=400)
            plt.savefig(path + "/latest.png", dpi=400)
            plt.close()

        iteration += 1

        mu_standard_obj, var_standard_obj = inference(gp, jnp.array([x_opt]))

        x_opt = list(x_opt)
        x_opt = [jnp.float64(xi) for xi in x_opt]
        print("Optimal Solution: ", x_opt)

        sample = sample_to_dict(x_opt, x_bounds)

        run_info = {
            "id": "running",
            "inputs": sample,
            "cost": "running",
            "objective": "running",
            "pred_obj_mean": np.float64(mu_standard_obj),
            "pred_obj_std": np.float64(np.sqrt(var_standard_obj)),
        }
        try:
            run_info["MSE"] = np.float64(error)
        except:
            print("No Error Calculation")
        data["data"].append(run_info)
        save_json(data, data_path)

        s_eval = sample.copy()
        res = f(s_eval)

        for k, v in res.items():
            run_info[k] = v

        run_info["objective"] = np.float64(run_info["objective"]).item()
        run_info["cost"] = np.float64(run_info["cost"]).item()

        data["data"][-1] = run_info
        save_json(data, data_path)

