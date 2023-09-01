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
    data_path,
    problem_data,
    path,
    printing=False,
    key=random.PRNGKey(0),
):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    sample_initial = problem_data["sample_initial"]
    gp_ms = problem_data["gp_ms"]

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
    alternatives = problem_data["alternatives"]

    save_json(data, data_path)

    iteration = len(data["data"]) - 1

    while True:
        start_time = time.time()
        data = read_json(data_path)
        inputs, outputs, cost = format_data(data)
        f_best = np.max(outputs)
        gp = build_gp_dict(*train_gp(inputs, outputs, gp_ms))
        util_args = (gp, f_best)

        n_test = 1000
        x_test = jnp.linspace(x_bounds["x"][0], x_bounds["x"][1], n_test).reshape(-1, 1)

        y_true = f.eval_vector(x_test[:, 0])

        aq = vmap(f_aq, in_axes=(0, None))
        aq_vals_list = aq(x_test, util_args)

        posterior = gp["posterior"]
        D = gp["D"]
        latent_dist = posterior.predict(x_test, train_data=D)
        predictive_dist = posterior.likelihood(latent_dist)
        mean = predictive_dist.mean()
        cov = jnp.sqrt(predictive_dist.variance())

        # optimising the aquisition of inputs, disregarding fidelity
        print("Optimising utility function...")

        # sample and normalise initial guesses

        n_opt = int(len(x_bounds.values()) * alternatives)
        upper_bounds = jnp.array([b[1] for b in list(x_bounds.values())])
        lower_bounds = jnp.array([b[0] for b in list(x_bounds.values())])
        upper_bounds = jnp.repeat(upper_bounds, alternatives)
        lower_bounds = jnp.repeat(lower_bounds, alternatives)

        termination = get_termination("n_gen", problem_data["NSGA_iters"])

        algorithm = NSGA2(
            pop_size=30,
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
                x_sols = jnp.split(x, alternatives, axis=1)
                K_list = []
                for i in range(len(x_sols[0])):
                    set = jnp.array([x_sols[j][i] for j in range(alternatives)])
                    K = gp["posterior"].prior.kernel.gram(set).matrix
                    K = jnp.linalg.det(K)
                    K_list.append(K)
                K_list = np.array(K_list)

                aq_list = np.sum([aq(x_i, util_args) for x_i in x_sols], axis=0)
                out["F"] = [aq_list, -K_list]

        problem = MO_aq()
        res = minimize(
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

        x_best_aq = X[np.argmin(AQ)]
        x_best_d = X[np.argmin(D)]
        x_best_utopia = X[np.argmin(distances)]

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        best_aq_sol = np.argmin(AQ)
        best_D_sol = np.argmin(D)
        utopia_sol = np.argmin(distances)

        ax.scatter(
            -F[:, 0],
            -F[:, 1],
            s=20,
            edgecolors="k",
            facecolors="None",
            alpha=0.5,
            label="Pareto Solutions",
        )
        ax.scatter(
            -F[best_aq_sol, 0],
            -F[best_aq_sol, 1],
            s=50,
            c="#FFC107",
            label="Best Acquisition Sum",
        )
        ax.scatter(
            -F[best_D_sol, 0],
            -F[best_D_sol, 1],
            s=50,
            c="#D81B60",
            label="Best Joint Variability",
        )
        ax.scatter(-F[utopia_sol, 0], -F[utopia_sol, 1], s=50, c="k", label="Utopia")
        ax.set_xlabel("Sum of Acquisition Function Values")
        ax.set_ylabel("Joint Variability")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # legend below plot
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        fig.tight_layout()
        fig.savefig(path + "/" + str(iteration + 1) + "_pareto.png", dpi=600)

        random_choice = np.random.randint(0, alternatives)  # should be made by a human
        x_opt = np.array([x_best_utopia[random_choice]])

        # x_opt = xs[jnp.argmin(jnp.array(aq_vals))]

        if printing == True:
            fig, axs = plt.subplots(2, 1, figsize=(8, 4))
            ax = axs[0]
            max_f = np.argmax(y_true)
            ax.plot(x_test[:, 0], y_true, c="k", lw=2, label="Function", alpha=0.5)
            ax.scatter(
                x_test[max_f],
                y_true[max_f],
                c="k",
                s=40,
                marker="+",
                label="Global Optimum",
            )
            ax.scatter(inputs, outputs, c="k", s=20, lw=0, label="Data")
            # remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("$x$")
            ax.set_ylabel("$f(x)$")
            ax.plot(x_test[:, 0], mean, c="k", ls="--", lw=2, label="GP Posterior")
            ax.fill_between(
                x_test[:, 0],
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
            ax.plot(
                x_test[:, 0],
                aq_vals_list,
                c="k",
                lw=2,
                label="Acquisition Function",
                zorder=-1,
            )
            ax.fill_between(
                x_test[:, 0],
                min(aq_vals_list),
                aq_vals_list,
                alpha=0.05,
                color="k",
                lw=0,
            )
            ax.set_xlabel("$x$")
            ax.set_ylabel("$\mathcal{U}(x)$")
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # ax.scatter(x_opt, -f_aq(x_opt,(gp)), c="k", s=20, lw=0, label="Optimum")
            for i in range(alternatives):
                ax.scatter(
                    x_best_aq[i],
                    -f_aq(x_best_aq[i], util_args),
                    c="#FFC107",
                    s=40,
                    label="Best Acquisition Set" if i == 0 else None,
                )
                ax.scatter(
                    x_best_d[i],
                    -f_aq(x_best_d[i], util_args),
                    c="#D81B60",
                    s=40,
                    label="Best Variability Set" if i == 0 else None,
                )
                ax.scatter(
                    x_best_utopia[i],
                    -f_aq(x_best_utopia[i], util_args),
                    c="k",
                    s=40,
                    label="Utopia Set" if i == 0 else None,
                )

            ax.legend(
                frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4
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
