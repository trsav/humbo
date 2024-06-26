from utils import *
import os
import jax.random as random
from jax import vmap
from jaxopt import ScipyBoundedMinimize as bounded_solver
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.dnsga2 import DNSGA2
import gc 
import datetime 
import resource
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
import uuid
from pymoo.optimize import minimize as minimize_mo
# from scipy.optimize import minimize as scipy_minimize
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.create_problem import * 
from function_creation.function import *


def log_variable_sizes(local_vars):
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in local_vars.items()), key=lambda x: -x[1]):
        print(f"Variable: {name}, Size: {size} bytes")


def bo(
    f,
    f_aq,
    problem_data,
):
    path = problem_data["file_name"]
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
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

    print(data)

    problem_data['f_opt'] = (f.f_opt)
    data["problem_data"] = problem_data
    alternatives = int(problem_data["alternatives"])
    save_json(data, data_path)

    iteration = len(data["data"]) - 1
    while len(data['data']) < problem_data['max_iterations']:
        
        if problem_data['plotting'] == True:
            try:
                os.mkdir(path + "/" + str(iteration + 1))
            except FileExistsError:
                pass
            
        start_time = time.time()
        data = read_json(data_path)
        inputs, outputs, cost = format_data(data)
        d = len(inputs[0])
        f_best = np.max(outputs)
        gp = build_gp_dict(*train_gp(inputs, outputs, gp_ms))
        util_args = (gp, f_best)


        aq = vmap(f_aq, in_axes=(0, None))
        if problem_data['dim'] == 1 and problem_data['plotting'] == True:
            n_test = 1000
            x_test = jnp.linspace(x_bounds[0][0], x_bounds[0][1], n_test).reshape(-1, 1)
            y_true = f.eval_vector(x_test[:, 0])
            aq_vals_list = aq(x_test, util_args)
            posterior = gp["posterior"]
            D = gp["D"]
            latent_dist = posterior.predict(x_test, train_data=D)
            predictive_dist = posterior.likelihood(latent_dist)
            mean = predictive_dist.mean()
            cov = jnp.sqrt(predictive_dist.variance())

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

            if problem_data['plotting'] == True:
                fig, ax = plt.subplots(1, 1, figsize=(7, 3.5))
                arg_sort = np.argsort(-F[:,0])
                ax.scatter(
                    -F[arg_sort, 0],
                    -F[arg_sort, 1],
                    c= 'k',
                    marker='+',
                    lw=1,
                    label='Pareto Solutions'
                )
                ax.scatter(
                    -F[best_aq_sol, 0],
                    -F[best_aq_sol, 1],
                    s=50,
                    c="#FFC107",
                    label="Best Utility Sum",
                )
                ax.scatter(
                    -F[best_D_sol, 0],
                    -F[best_D_sol, 1],
                    s=50,
                    c="#D81B60",
                    label="Best Joint Variability",
                )
                ax.scatter(-F[utopia_sol, 0], -F[utopia_sol, 1], s=50, c="k", label="Knee-solution")
                ax.set_xlabel("Sum of Utility Function Values")
                ax.set_ylabel("Joint Variability")
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                # legend below plot
                ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
                fig.tight_layout()
                fig.savefig(path + "/" + str(iteration + 1) + "/pareto.pdf")
                plt.close()

            if problem_data['plotting'] == True:


                fig,axs = plt.subplots(1,alternatives+1,figsize=(10,4))
                for i in range(len(axs)-2):
                    axs[i].get_shared_y_axes().joined(axs[i], axs[i+1])
                for i in range(len(axs)-1):
                    m,sigma = inference(gp, jnp.array([x_best_utopia[i]]))
                    sigma = np.sqrt(sigma)
                    p_y = tfd.Normal(loc=m, scale=sigma)
                    y = np.linspace(m-3*sigma,m+3*sigma,100)
                    p_y_vals = p_y.prob(y)
                    for j in range(len(axs)-1):
                        axs[j].fill_betweenx(y[:,0],[0 for i in range(100)],p_y_vals[:,0],alpha=0.05,color='k')
                    axs[i].plot(p_y_vals,y,c='k',lw=1)
                    axs[i].fill_betweenx(y[:,0],[0 for i in range(100)],p_y_vals[:,0],alpha=0.2,color='k')
                    axs[i].plot([0,p_y.prob(m)[0]],[m,m],c='k',lw=1,ls='--')
                    axs[i].set_title('Choice ' + str(i+1))
                    axs[i].set_xlabel(r"$p(f(x))$")

                axs[0].set_ylabel(r"$f(x)$")
                bar_labels = [str(i+1) for i in range(alternatives)]
                aq_vals = [-aq(jnp.array([x_best_utopia[i]]), util_args).item() for i in range(alternatives)]
                cols = ['k' for i in range(alternatives)]
                axs[-1].bar(bar_labels,aq_vals,color=cols,alpha=0.5,edgecolor='k',lw=1)
                axs[-1].set_ylabel(r"$\mathcal{U}(x)$")
                axs[-1].set_xlabel("Choices")


                for ax in axs:
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                fig.tight_layout()
                fig.savefig(path + "/" + str(iteration + 1) + "/choices.pdf")
                plt.close() 
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
        if problem_data['plotting'] == True and problem_data['dim'] == 1:
            fig, axs = plt.subplots(2, 1, figsize=(8, 4))
            ax = axs[0]
            max_f = np.argmax(y_true)
            ax.plot(x_test[:, 0], y_true, c="k", lw=1, label="Function", alpha=0.5)
            ax.scatter(
                x_test[max_f],
                y_true[max_f],
                c="k",
                s=40,
                marker="*",
                label="Global Optimum",
            )
            ax.scatter(inputs, outputs, c="k", s=20, lw=0, label="Data")
            # remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xticks([])
            ax.set_xlabel("$x$")
            ax.set_ylabel("$f(x)$")
            ax.plot(x_test[:, 0], mean, c="k", ls="--", lw=1, label="GP Posterior")
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
                lw=1,
                label="Utility Function",
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

            if problem_data['human_behaviour'] != 'trusting':
                for i in range(alternatives-1):
                    ax.scatter(
                        x_best_d[i],
                        -f_aq(x_best_d[i], util_args),
                        c="k",
                        marker="+",
                        s=40,
                        label="Best Variability Set" if i == 0 else None,
                    )
                    ax.scatter(
                        x_best_utopia[i],
                        -f_aq(x_best_utopia[i], util_args),
                        c="k",
                        s=40,
                        marker='^',
                        label="Knee Set" if i == 0 else None,
                    )
                    ax.text(
                    x_best_utopia[i],
                    -f_aq(x_best_utopia[i], util_args) + 0.25,
                    "Choice "+str(i+1),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )


            ax.scatter(
                x_opt_aq,
                -f_aq(x_opt_aq, util_args),
                c="k",
                s=40,
                label='Optimum'
            )
 
            # text of 'choice x' about this scatter
            ax.text(
                x_opt_aq,
                -f_aq(x_opt_aq, util_args) + 0.25,
                "Choice "+str(alternatives),
                ha="center",
                va="bottom",
                fontsize=8,
            )




            u_opt = -f_aq(x_opt, util_args).item()
            ax.plot([x_opt, x_opt], [u_opt, min(aq_vals_list)], c="k", lw=1, ls="--",label='Selected')

            ax.legend(
                frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=5,fontsize=8
            )
            fig.tight_layout()
            plt.savefig(path + "/" + str(iteration + 1) + "/utility.pdf")
            plt.savefig(path + "/latest.pdf")
            plt.close()

        iteration += 1

        mu_opt,var_opt = inference(gp, jnp.array([x_opt]))


        if d > 1 and problem_data['human_behaviour'] != 'trusting':
            x_opt = x_opt[0]

        if d > 1 and problem_data['human_behaviour'] == 'trusting':

            x_opt = [float(x) for  x in x_opt]
        print(x_opt)
        
        

        print("Optimal Solution: ", x_opt)


        run_info = {
            "id": "running",
            "inputs": list(x_opt),
            "pred_mu": np.float64(mu_opt),
            "pred_sigma": np.float64(np.sqrt(var_opt)),
        }
        
        data["data"].append(run_info)
        save_json(data, data_path)

        f_eval =  f(x_opt)
        run_info["objective"] = f_eval
        run_info["id"] = str(uuid.uuid4())

        # if value is array then get item if not doesn't matter 
        regret = (f.f_opt - max(f_eval,jnp.max(outputs)))
        if regret.__class__ != float:
            regret = regret.item()
        run_info["regret"] = regret


        data["data"][-1] = run_info
        print(data)
        save_json(data, data_path)

        if problem_data['plotting'] == True:

            regret_list = [d['regret'] for d in data['data']]
            init = problem_data['sample_initial']
            it = len(regret_list)
            fig,axs = plt.subplots(1,2,figsize=(10,4))
            fs = 16
            ax = axs[0]
            ax.plot(np.arange(it),regret_list,c='k',lw=1)
            axs[0].set_ylabel(r"$r_\tau$",fontsize=fs)
            axs[0].set_xlabel(r"$\tau$",fontsize=fs)
            axs[1].set_ylabel(r"$\frac{R_\tau}{\tau}$",fontsize=fs)
            obj = [d['objective'] for d in data['data']]
            cumulative_regret = [t*f.f_opt - np.sum(obj[:t]) for t in range(1,it+1)]
            average_regret = [f.f_opt - (1/t) * np.sum(obj[:t]) for t in range(1,it+1)]

            ax = axs[1]
            ax.set_xlabel(r"$\tau$",fontsize=fs)

            ax.plot(np.arange(it),average_regret,c='k',lw=1)
            ax.plot([0,it],[0,0],c='k',lw=1,ls='--',label='Reference')

            ax.legend(frameon=False)

            for ax in axs:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig.savefig(path + "/regret.pdf")
            plt.close()


# aq = 'UCB'
# problem_data = {}
# problem_data["sample_initial"] = 2
# problem_data["gp_ms"] = 16
# problem_data["alternatives"] = 4
# problem_data["NSGA_iters"] = 200
# problem_data['deterministic_initial'] = 'true'
# problem_data['max_iterations'] = 60
# problem_data['plotting'] = True
# problem_data['acquisition_function'] = aq
# problem_data['time_created'] = str(datetime.datetime.now())

# problem_data['dim'] = 1
# problem_data['human_behaviour'] = 'expert'

# k = jax.random.PRNGKey(10)
# f = Function(create_problem(k,0.04,problem_data['dim']))

# file = "1D_vis_"
# problem_data['file_name'] = "bo/plots/" + file
# problem_data['function'] = file

# aqs = {'UCB':UCB,'EI':EI}
# bo(f, aqs[aq], problem_data)


### Augmented LHS 
            

# bounds = np.array([[0,1],[0,1]])
# x_expert = np.array([[1,0],[0,1],[0.5,0.5],[0.25,0.25],[0.25,0.75]])

# n = len(x_expert)
# m = 24
# required = m - n 




# def k(x1,x2):
#     return np.exp(-np.sum((x1-x2)**2)/0.1)

# def covar_mat(x):
#     n = len(x)
#     K = np.zeros((n,n))
#     for i in range(n):
#         for j in range(i,n):
#             K[i,j] = k(x[i],x[j])
#             K[j,i] = K[i,j]
#     return K

# def obj(x_list,x_expert):
#     x_random = np.reshape(x_list,(required,2))
#     x_aug = np.concatenate((x_random,x_expert),axis=0)
#     K = covar_mat(x_aug)
#     det = np.linalg.det(K)
#     return -det

# import scipy.optimize as opt

# x_list = np.random.uniform(size=(required*2))

# print('Running Nelder-Mead')
# res = opt.minimize(obj,x_list,args=(x_expert),method='Nelder-Mead',bounds=[(0,1) for i in range(required*2)],options={'disp':True,'maxiter':10000})
# x_opt = res.x
# res = opt.minimize(obj,x_opt,args=(x_expert),method='CG',bounds=[(0,1) for i in range(required*2)],options={'disp':True,'maxiter':10000})
# x_opt = res.x
# x_opt = np.reshape(x_opt,(required,2))
# x_list = np.reshape(x_list,(required,2))

# fig,axs = plt.subplots(1,3,figsize=(11,4))

# axs[1].scatter(x_list[:,0],x_list[:,1],c='k',s=30,marker='o',edgecolor='k',label='Initial Solutions')
# axs[2].scatter(x_opt[:,0],x_opt[:,1],c="#D81B60",edgecolor="#D81B60",s=30,marker='o',label='Optimal Initial Solutions')
# for i in range(len(axs)):
#     ax = axs[i]
#     # maintain perspective 
#     ax.set_aspect('equal')
#     ax.scatter(x_expert[:,0],x_expert[:,1],edgecolor="#FFC107",c="#FFC107",s=30,marker='o',label='Expert Solutions' if i == 0 else None)
#     # draw box for bounds 
#     ax.plot([0,1],[0,0],c='k',lw=1,ls='--',alpha=0.5,label='Bounds' if i == 0 else None)
#     ax.plot([0,0],[0,1],c='k',lw=1,ls='--',alpha=0.5)
#     ax.plot([1,1],[0,1],c='k',lw=1,ls='--',alpha=0.5)
#     ax.plot([0,1],[1,1],c='k',lw=1,ls='--',alpha=0.5)

#     ax.set_xlim([-0.25,1.25])
#     ax.set_ylim([-0.25,1.25])
#     ax.set_xlabel(r"$x_1$",fontsize=16)
#     ax.set_ylabel(r"$x_2$",fontsize=16)
#     ax.set_xticks([])
#     ax.set_yticks([])

# lines, labels = axs[0].get_legend_handles_labels()
# lines2, labels2 = axs[1].get_legend_handles_labels()
# lines3, labels3 = axs[2].get_legend_handles_labels()
# lines = lines + lines2 + lines3
# labels = labels + labels2 + labels3
# fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.88), ncol=4,frameon=False,fontsize=16)

# fig.tight_layout()

# fig.savefig('lhs_fig.pdf')
                