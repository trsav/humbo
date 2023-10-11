from jax.config import config
import pandas as pd 
from jax import numpy as jnp
from jax import jit, value_and_grad, vmap
import matplotlib
import jax.random as random
from pymoo.core.problem import ElementwiseProblem, Problem
import os
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
import numpy as np
import numpy.random as rnd
import jax
import gpjax as gpx
import optax as ox
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from jax.scipy.optimize import minimize
from tensorflow_probability.substrates import jax as tfp
import shutil 
from matplotlib import rc
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

tfd = tfp.distributions


def sample_function(xb, z, eval):
    x_sample = {}
    x_sample["z1"] = z
    y = []
    c = []
    x = np.linspace(xb[0], xb[1], 300)
    for xi in x:
        x_sample["x1"] = xi
        res = eval(x_sample)
        y.append(res["objective"])
        c.append(res["cost"])
    return x, y, c


def plot_toy(eval, path, x_bounds, z_bounds):
    cmap = matplotlib.cm.get_cmap("Spectral")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    xb = x_bounds["x1"]
    x, y, c = sample_function(xb, z_bounds["z1"][1], eval)
    ax[0].plot(
        x, y, c="k", lw=3, label=r"$f(x,z=1):= \frac{2\cos(x)}{x} + \frac{\sin(3x)}{2}$"
    )
    ax[1].plot(x, c, c="k", lw=3, label="$c(x,z=1)$")
    x, y, c = sample_function(xb, z_bounds["z1"][0], eval)
    ax[0].plot(x, y, c="tab:red", lw=3, label=r"$f(x,z=0):= \sin(x)$")
    ax[1].plot(x, c, c="tab:red", lw=3, label="$c(x,z=0)$")
    ax[1].legend(frameon=False, fontsize=8)
    n = 10
    rgbr = [214, 39, 40]
    rgbb = [0, 0, 0]
    for i in np.linspace(0, 1, n):
        col = np.array([i * rgbr[j] + (1 - i) * rgbb[j] for j in range(3)]) / 256
        x, y, c = sample_function(
            xb, i * z_bounds["z1"][0] + (1 - i) * z_bounds["z1"][1], eval
        )
        ax[0].plot(x, y, c=tuple(col), lw=3, alpha=0.2)
        ax[1].plot(x, c, c=tuple(col), lw=3, alpha=0.2)

    ax[0].set_xlabel("x", fontsize=14)
    ax[0].set_ylabel("f(x)", fontsize=14)
    ax[1].set_xlabel("x", fontsize=14)
    ax[1].set_ylabel("$c(x)$", fontsize=14)

    for axs in ax:
        axs.spines["top"].set_visible(False)
        axs.spines["right"].set_visible(False)
        axs.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.savefig(path + "vis.png", dpi=300)
    plt.close()
    return


def format_data(data):
    # Reads a data file and returns inputs, outputs, costs

    # initialise lists
    inputs = []
    obj = []
    cost = []
    # iterate over data
    for d in data["data"]:
        # check the solution isn't still running
        if d["id"] != "running":
            # append values to lists
            inputs += [list(d["inputs"])]
            obj += [d["objective"]]

    # reformat lists to correct shaped arrays
    inputs = jnp.array(inputs)
    obj = jnp.array(obj).reshape(-1, 1)
    cost = jnp.array(cost).reshape(-1, 1)

    return inputs, obj, cost


def save_json(data, path):
    # save a dictionary as a json file
    with open(path, "w") as f:
        json.dump(data, f)
    return


def sample_bounds(bounds, n):
    sample = lhs(jnp.array(bounds), n)
    return sample

def random_sample(bounds,n):
    sample = np.zeros((n,len(bounds)))
    for i in range(len(bounds)):
        sample[:,i] = np.random.uniform(bounds[i,0],bounds[i,1],n)
    return sample

def sample_to_dict(sample, bounds):
    # convert a list of values to a dictionary
    # using respective bounds keys

    sample_dict = {}
    keys = list(bounds.keys())
    for i in range(len(sample)):
        sample_dict[keys[i]] = float(sample[i])
    return sample_dict


def read_json(path):
    # read a json file as a dictionary
    with open(path, "r") as f:
        data = json.load(f)
    # close file
    return data


def numpy_lhs(bounds: list, p: int):
    d = len(bounds)

    sample = []
    for i in range(0, d):
        s = np.linspace(bounds[i, 0], bounds[i, 1], p)
        rnd.shuffle(s)
        sample.append(s)
    sample = np.array(sample).T
    return sample


def lhs(bounds: list, p: int):
    d = len(bounds)

    sample = []
    for i in range(0, d):
        s = jnp.linspace(bounds[i, 0], bounds[i, 1], p)
        s = jax.random.shuffle(jax.random.PRNGKey(np.random.randint(0,1000)), s)
        sample.append(s)
    sample = jnp.array(sample).T

    return sample

def plot_function(f,path):
    x = jnp.linspace(f.bounds["x"][0], f.bounds["x"][1], 500)
    y = f.eval_vector(x)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(x, y, c="k", lw=1)
    x_opt = x[jnp.argmax(y)]
    y_opt = jnp.max(y)

    ax.scatter(x_opt,y_opt, c="k",marker='+', s=50,label='Global Optimum')
    # plotting a line from the optimum to the x-axis
    ax.plot([x_opt,x_opt],[y_opt,jnp.min(y)], c="k", lw=1,linestyle='--',alpha=0.5)
    ax.plot([f.bounds["x"][0],f.bounds["x"][1]],[y_opt,y_opt], c="k", lw=1,linestyle='--',alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.savefig(path)
    return

def train_gp(inputs, outputs, ms):
    # creating a set of initial GP hyper parameters (log-spaced)
    p_num = len(inputs[0, :])

    init_params = np.zeros((ms, p_num))
    for i in range(0, p_num):
        init_params[:, i] = np.geomspace(0.1, 1, ms)
    rnd.shuffle(init_params[:, i])
    init_params = jnp.array(init_params)
    # defining dataset
    D = gpx.Dataset(X=inputs, y=outputs)
    # for each intital list of hyperparameters
    best_nll = 1e30

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, ms)
    nlls = []
    opt_posteriors = []

    for p in init_params:
        kernel = gpx.kernels.Matern52(lengthscale=p)
        meanf = gpx.mean_functions.Constant()
        prior = gpx.Prior(mean_function=meanf, kernel=kernel)
        likelihood = gpx.Gaussian(num_datapoints=D.n, obs_noise=0.0)
        # Bayes rule
        posterior = prior * likelihood
        # negative log likelihood
        negative_mll = gpx.objectives.ConjugateMLL(negative=True)
        negative_mll = jit(negative_mll)

        opt_posterior, history = gpx.fit(
            model=posterior,
            objective=negative_mll,
            train_data=D,
            optim=ox.adam(learning_rate=0.01),
            num_iters=2000,
            safe=True,
            key=key,
        )

        nll = float(history[-1])
        nlls.append(nll)
        opt_posteriors.append(opt_posterior)

    best_posterior = opt_posteriors[np.argmax(nlls)]
    return best_posterior, D


def inference(gp, inputs):
    posterior = gp["posterior"]
    D = gp["D"]

    latent_dist = posterior.predict(inputs, train_data=D)
    predictive_dist = posterior.likelihood(latent_dist)

    predictive_mean = predictive_dist.mean().astype(float)
    predictive_std = predictive_dist.stddev().astype(float)
    return predictive_mean, predictive_std


def build_gp_dict(posterior, D):
    # build a dictionary to store features to make everything cleaner
    gp_dict = {}
    gp_dict["posterior"] = posterior
    gp_dict["D"] = D
    return gp_dict


@jit
def calculate_entropy_sample(x, x_s, y_s, l_s, gp):
    D = gp["D"] + gpx.Dataset(X=x, y=jnp.array([y_s]))
    kernel = gpx.kernels.Matern52(lengthscale=l_s)
    meanf = gpx.mean_functions.Constant()
    prior = gpx.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.Gaussian(num_datapoints=D.n, obs_noise=0.0)
    posterior = prior * likelihood
    gp_s = build_gp_dict(posterior, D)
    m_s, v_s = inference(gp_s, jnp.array([x_s]))
    return v_s[0]


def gaussian_differential_entropy(K, N):

    return 0.5 * jnp.log((2 * np.pi * jnp.exp(1)) ** N * jnp.linalg.det(K))


def global_optimum_distributions(x_bounds, gp, samples):
    d = len(x_bounds)
    gp_sample_size = 20**d
    x_s = lhs(jnp.array(list(x_bounds.values())), gp_sample_size)
    K = gp["posterior"].prior.kernel.gram(x_s).matrix
    mu = gp["posterior"].prior.mean_function(x_s)[:, 0]
    key = jax.random.PRNGKey(0)
    x_s = x_s[:, 0]
    m_s = random.multivariate_normal(key, mean=mu, cov=K, shape=(samples, 1))[:, 0, :]
    x_samples = x_s[m_s.argmax(axis=1)]
    f_samples = m_s.max(axis=1)
    return x_samples, f_samples

@jit
def EI(x, args):
    gp, f_best = args
    m, K = inference(gp, jnp.array([x]))
    sigma = jnp.sqrt(K)
    diff = m - f_best
    p_y = tfd.Normal(loc=m, scale=sigma)
    Z = diff / sigma
    expected_improvement = sigma*(Z * p_y.cdf(Z) +p_y.prob(Z))
    return - expected_improvement[0]

@jit
def UCB(x, args):
    gp, f_best = args
    m, K = inference(gp, jnp.array([x]))
    sigma = jnp.sqrt(K)
    return -(m + 3*sigma)[0]





def plot_regret(problem_data,axs,c,directory,max_it):
    
    files = os.listdir(directory)
    problem_data_list = []
    for i in tqdm(range(len(files))):
        if '.' not in files[i] and '_' not in files[i]:
            results = directory+'/'+files[i] + '/res.json'
            # open json
            with open(results, "r") as f:
                data = json.load(f)
            problem_data_list.append(data['problem_data'])

    # create dataframe from list of dictionaries 
    df = pd.DataFrame(problem_data_list)
    df = df.loc[(df['human_behaviour'] == problem_data['human_behaviour']) & (df['acquisition_function'] == problem_data['acquisition_function']) & (df['dim'] == problem_data['dim'])]

    file_names = df['file_name'].values
    regret_list = []
    obj_list = []
    f_opt_list = []
    for file in file_names:
        file = file.split('results/')[1]
        file = directory + '/' +  file 
        data_full = read_json(file+'res.json')
        data = data_full['data']
        f_opt = data_full['problem_data']['f_opt']
        obj = [d['objective'] for d in data]
        f_opt_list.append(f_opt)
        obj_list.append(obj)

    init = problem_data['sample_initial']
    full_it = problem_data['max_iterations']

    average_regret_list = []
    regret_list = []
    for obj,i in zip(obj_list,range(len(obj_list))):
        if len(obj) != full_it:
            obj += [obj[-1]]*(full_it-len(obj))

        it = len(obj)
        regret = [f_opt_list[i] - np.max(obj[:t]) for t in range(1,it+1)]
        regret_list.append(regret)
        cumulative_regret = [f_opt_list[i] - np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret = [f_opt_list[i] - (1/t) * np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret_list.append(average_regret)

    regret_list = np.array(regret_list)[:,:max_it]
    average_regret = np.mean(np.array(average_regret_list),axis=0)[:max_it]
    average_regret_std = np.std(np.array(average_regret_list),axis=0)[:max_it]
    label = problem_data['human_behaviour']

    if label.__class__ != float:
        label = label[0].upper() + label[1:]
    else:
        label = '$p($Best$)=$'+str(label) 
    # label = '$\mathbb{E}[$'+label+'$]$'
    # captialise first letter 

    try:
        x = np.arange(init,len(average_regret))
        axs[1].plot(x,average_regret[init:],c=c,lw=1.5,label=label)
        axs[1].fill_between(x,average_regret[init:]-average_regret_std[init:],average_regret[init:]+average_regret_std[init:],alpha=0.1,color=c,lw=0)


        ax = axs[0]
        regret_list = np.array(regret_list)
        mean_instantaneous_regret = np.mean(regret_list,axis=0)
        std_instantaneous_regret = np.std(regret_list,axis=0)
        x = np.arange(init,len(mean_instantaneous_regret))
        ax.plot(x,mean_instantaneous_regret[init:],c=c,lw=1.5,label=label)
        lower = mean_instantaneous_regret[init:]-std_instantaneous_regret[init:]
        # cut of less than 0
        lower[lower<0] = 0
        upper = mean_instantaneous_regret[init:]+std_instantaneous_regret[init:]
        ax.fill_between(x,lower,upper,alpha=0.1,color=c,lw=0)
    except:
        return 
    return 

def delete_folders(problem_data):
    directory = 'bo/benchmark_results_specific'
    files = os.listdir(directory)
    problem_data_list = []
    for i in tqdm(range(len(files))):
        if '.' not in files[i] and '_' not in files[i]:
            results = directory+'/'+files[i] + '/res.json'
            # open json
            with open(results, "r") as f:
                data = json.load(f)
            problem_data_list.append(data['problem_data'])

    # create dataframe from list of dictionaries 
    df = pd.DataFrame(problem_data_list)

    df = df.loc[(df['function'] == problem_data['function']) & (df['acquisition_function'] == problem_data['acquisition_function'])]

    file_names = df['file_name'].values
    regret_list = []
    obj_list = []
    f_opt_list = []
    for file in file_names:
        shutil.rmtree(file)
    return 

# for d in [1,2,5]:
#     problem_data = {}
#     problem_data['algorithm'] = 'random'
#     problem_data["batch_size"] = 8
#     problem_data["gp_ms"] = 8
#     problem_data["NSGA_iters"] = 75
#     problem_data["plotting"] = False
#     problem_data['max_batches'] = 50
#     problem_data['lengthscale'] = 0.3
#     #problem_data['lengthscale'] = 0.8
#     problem_data['dim'] = d
#     # at a given human behaviour
#     problem_data['acquisition_function'] = 'UCB'
#     delete_folders(problem_data)


# f_store = ['Branin']
# for i in [2,5,10]:
#     f_store.append('Ackley'+str(i))
#     f_store.append('Griewank'+str(i))
#     f_store.append('Rastrigin'+str(i))
#     f_store.append('Rosenbrock'+str(i))
#     f_store.append('Powell'+str(i))

# for f in f_store:
#     problem_data = {}
#     problem_data['function'] = f
#     problem_data["sample_initial"] = 4
#     problem_data["gp_ms"] = 8
#     problem_data["alternatives"] = 4
#     problem_data["NSGA_iters"] = 100
#     problem_data["plotting"] = False
#     problem_data['max_iterations'] = 100
#     problem_data['human_behaviour'] = 'trusting'
#     problem_data['acquisition_function'] = 'UCB'
#     delete_folders(problem_data)

def plot_regret_specific(problem_data,axs,c,directory,function):
    
    files = os.listdir(directory)
    problem_data_list = []
    for i in tqdm(range(len(files))):
        if '.' not in files[i] and files[i].split('_')[0] == function:
            results = directory+'/'+files[i] + '/res.json'
            # open json
            with open(results, "r") as f:
                data = json.load(f)
            problem_data_list.append(data['problem_data'])

    # create dataframe from list of dictionaries 
    df = pd.DataFrame(problem_data_list)
    df = df.loc[(df['human_behaviour'] == problem_data['human_behaviour']) & (df['acquisition_function'] == problem_data['acquisition_function'])]

    file_names = df['file_name'].values
    regret_list = []
    obj_list = []
    f_opt_list = []
    for file in file_names:
        try:
            file = file.split('results_specific/')[1]
        except:
            file = file.split('results_specific_trusting/')[1]
        file = directory + '/' +  file 
        data_full = read_json(file+'res.json')
        data = data_full['data']
        f_opt = data_full['problem_data']['f_opt']
        try:
            obj = [d['objective'] for d in data]
            f_opt_list.append(f_opt)
            obj_list.append(obj)
        except:
            pass

    init = problem_data['sample_initial']
    full_it = problem_data['max_iterations']

    average_regret_list = []
    regret_list = []
    for obj,i in zip(obj_list,range(len(obj_list))):
        if len(obj) != full_it:
            obj += [obj[-1]]*(full_it-len(obj))

        it = len(obj)
        regret = [f_opt_list[i] - np.max(obj[:t]) for t in range(1,it+1)]
        regret_list.append(regret)
        cumulative_regret = [f_opt_list[i] - np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret = [f_opt_list[i] - (1/t) * np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret_list.append(average_regret)

    n = 100 
    average_regret = np.mean(np.array(average_regret_list),axis=0)[:n]
    average_regret_std = np.std(np.array(average_regret_list),axis=0)[:n]
    regret_list = np.array(regret_list)[:,:n]
    label = problem_data['human_behaviour']

    if label.__class__ != float:
        label = label[0].upper() + label[1:]
    else:
        label = '$p($Best$)=$'+str(label) 
    # label = '$\mathbb{E}[$'+label+'$]$'
    # captialise first letter 

    try:
        x = np.arange(init,len(average_regret))

        axs[1].plot(x,average_regret[init:],c=c,lw=1.5,label=label)
        #axs[1].fill_between(x,average_regret[init:]-average_regret_std[init:],average_regret[init:]+average_regret_std[init:],alpha=0.1,color=c,lw=0)


        ax = axs[0]
        regret_list = np.array(regret_list)
        mean_instantaneous_regret = np.mean(regret_list,axis=0)
        std_instantaneous_regret = np.std(regret_list,axis=0)
        x = np.arange(init,len(mean_instantaneous_regret))
        ax.plot(x,mean_instantaneous_regret[init:],c=c,lw=1.5,label=label)
        lower = mean_instantaneous_regret[init:]-std_instantaneous_regret[init:]
        # cut of less than 0
        lower[lower<0] = 0
        upper = mean_instantaneous_regret[init:]+std_instantaneous_regret[init:]
        #ax.fill_between(x,lower,upper,alpha=0.1,color=c,lw=0)
    except:
        return 
    return 



def plot_human(aq,d,max_it):
    directory = 'bo/benchmark_results'
    colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
    human_behaviours = ['expert','adversarial','trusting',0.75,0.5,0.25]


    fig,axs = plt.subplots(1,2,figsize=(8,2.5))

    for i in range(len(human_behaviours)):
        # for this problem data
        problem_data = {}
        problem_data["sample_initial"] = 4
        problem_data["gp_ms"] = 8
        problem_data["alternatives"] = 4
        problem_data["NSGA_iters"] = 75
        problem_data["plotting"] = True
        problem_data['max_iterations'] = 150
        problem_data['lengthscale'] = 0.3
        #problem_data['lengthscale'] = 0.8
        problem_data['dim'] = d
        # at a given human behaviour
        problem_data['human_behaviour'] = human_behaviours[i]
        problem_data['acquisition_function'] = aq

        plot_regret(problem_data,axs,colors[i],directory,max_it)
    fs = 12
    axs[0].set_ylabel(r"Simple Regret, $r_\tau$",fontsize=fs)
    for ax in axs:
        ax.grid(True,alpha=0.5)
        x_start = problem_data['sample_initial']
        max_y = ax.get_ylim()[1]
        min_y = ax.get_ylim()[0]
        ax.plot([x_start,x_start],[min_y,max_y],c='k',ls='--',lw=1,alpha=0.5)
    axs[0].set_xlabel(r"Iterations, $\tau$",fontsize=fs)
    axs[1].set_xlabel(r"Iterations, $\tau$",fontsize=fs)
    axs[1].set_ylabel(r"Average Regret, ${R_\tau}/{\tau}$",fontsize=fs)
    # add text in upper right of right plot with functon name

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.875), ncol=6,frameon=False)

    l = problem_data['lengthscale']

    #fig.suptitle(r'Regret expectation over 50 functions, $f \sim \mathcal{GP}(\mu \equiv 0, K_M (d,\nu = '+str(l)+'))$, '+str(problem_data['alternatives'])+' alternate choices, $\mathcal{U}(x)=$'+str(aq)+r', $x \in R^'+str(problem_data['dim'])+'$',fontsize=int(fs))
    fig.tight_layout()
    fig.subplots_adjust(top = 0.875)
    plt.savefig('bo/plots/overall_regret_aq_'+str(aq)+'_d_'+str(d)+'.pdf')

# plot_human('EI',1)
# plot_human('UCB',1,60)
# plot_human('UCB',2,80)
# plot_human('UCB',5,140)

def plot_human_specific():
    directory = 'bo/benchmark_results_specific'
    colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
    human_behaviours = ['expert','adversarial','trusting',0.75,0.5,0.25]


    functions = ['Branin']
    for i in [2,5,10]:
        functions.append('Ackley'+str(i))
        functions.append('Griewank'+str(i))
        functions.append('Powell'+str(i))
        functions.append('Rastrigin'+str(i))
        functions.append('Rosenbrock'+str(i))

    for function in functions:
        fig,axs = plt.subplots(1,2,figsize=(8,2.5))
        

        for i in range(len(human_behaviours)):
            # for this problem data
            problem_data = {}
            problem_data["sample_initial"] = 4
            problem_data["gp_ms"] = 8
            problem_data["alternatives"] = 4
            problem_data["NSGA_iters"] = 150
            problem_data["plotting"] = True
            problem_data['max_iterations'] = 150
            problem_data['human_behaviour'] = human_behaviours[i]
            problem_data['acquisition_function'] = 'UCB'

            plot_regret_specific(problem_data,axs,colors[i],directory,function)
        fs = 12
        axs[0].set_ylabel(r"Simple Regret, $r_\tau$",fontsize=fs)
        for ax in axs:
            ax.grid(True,alpha=0.5)
            x_start = problem_data['sample_initial']
            max_y = ax.get_ylim()[1]
            min_y = ax.get_ylim()[0]
            ax.plot([x_start,x_start],[min_y,max_y],c='k',ls='--',lw=1,alpha=0.5)
        axs[0].set_xlabel(r"Iterations, $\tau$",fontsize=fs)
        axs[1].set_xlabel(r"Iterations, $\tau$",fontsize=fs)
        axs[1].set_ylabel(r"Average Regret, ${R_\tau}/{\tau}$",fontsize=fs)
        # split function from number at end (2,5 or 10)
        if function.split('1')[-1] == '0':
            n = str(10)
            func_name = function.split('1')[0]
        else:
            n = function[-1]
            func_name = function[:-1]
        

        # text with white background in upper right of right plot with functon name

        axs[1].text(0.95, 0.95, func_name + ': $d= $'+n, horizontalalignment='right',verticalalignment='top', transform=axs[1].transAxes,fontsize=fs,bbox=dict(facecolor='white',edgecolor='none',pad=0.5))
        lines, labels = axs[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.875), ncol=6,frameon=False)

        fig.tight_layout()
        fig.subplots_adjust(top = 0.875,left = 0.125)
        

        axs[0].set_yscale('log')


        plt.savefig('bo/plots/overall_regret_'+function+'.pdf')

# plot_human_specific()

def plot_regret_batch(problem_data,axs,c,directory):
    
    files = os.listdir(directory)
    problem_data_list = []
    for i in tqdm(range(len(files))):
        if '.' not in files[i] and '_' not in files[i]:
            results = directory+'/'+files[i] + '/res.json'
            # open json
            with open(results, "r") as f:
                data = json.load(f)
            problem_data_list.append(data['problem_data'])

    # create dataframe from list of dictionaries 
    df = pd.DataFrame(problem_data_list)
    df = df.loc[(df['algorithm'] == problem_data['algorithm']) & (df['acquisition_function'] == problem_data['acquisition_function']) & (df['lengthscale'] == problem_data['lengthscale']) & (df['dim'] == problem_data['dim'])]

    file_names = df['file_name'].values
    regret_list = []
    obj_list = []
    f_opt_list = []
    for file in file_names:
        file = file.split('results/')[1]
        file = directory + '/' +  file 
        data_full = read_json(file+'res.json')
        data = data_full['data']
        f_opt = data_full['problem_data']['f_opt']
        obj = [d['objective'] for d in data]
        f_opt_list.append(f_opt)
        obj_list.append(obj)

    init = problem_data['batch_size']
    full_it = problem_data['max_batches'] * problem_data['batch_size']

    average_regret_list = []
    regret_list = []
    for obj,i in zip(obj_list,range(len(obj_list))):
        if len(obj) != full_it:
            obj += [obj[-1]]*(full_it-len(obj))

        it = len(obj)
        regret = [f_opt_list[i] - np.max(obj[:t]) for t in range(1,it+1)]
        regret_list.append(regret)
        cumulative_regret = [f_opt_list[i] - np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret = [f_opt_list[i] - (1/t) * np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret_list.append(average_regret)

    average_regret = np.mean(np.array(average_regret_list),axis=0)
    average_regret_std = np.std(np.array(average_regret_list),axis=0)
    label = problem_data['algorithm']

    label = label[0].upper() + label[1:]

    try:
        x = np.arange(init,len(average_regret))
        print(x)
        x = x / problem_data['batch_size']
        print(x)
        ax = axs
        regret_list = np.array(regret_list)
        mean_instantaneous_regret = np.mean(regret_list,axis=0)
        std_instantaneous_regret = np.std(regret_list,axis=0)
        x = np.arange(init,len(mean_instantaneous_regret))
        ax.plot(x,mean_instantaneous_regret[init:],c=c,lw=1.5,label=label)
        lower = mean_instantaneous_regret[init:]-std_instantaneous_regret[init:]
        # cut of less than 0
        lower[lower<0] = 0
        upper = mean_instantaneous_regret[init:]+std_instantaneous_regret[init:]
        ax.fill_between(x,lower,upper,alpha=0.1,color=c,lw=0)
    except:
        return 
    return 

def plot_batch(d):
    directory = 'bo/batch_benchmark_results'
    colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
    algorithms = ['random','batch']


    fig,ax = plt.subplots(1,1,figsize=(5,3.5))

    for i in range(len(algorithms)):
        # for this problem data
        problem_data = {}
        problem_data["batch_size"] = 8
        problem_data["gp_ms"] = 8
        problem_data["alternatives"] = 4
        problem_data["NSGA_iters"] = 200
        problem_data["plotting"] = True
        problem_data['max_batches'] = 50
        problem_data['lengthscale'] = 0.3
        #problem_data['lengthscale'] = 0.8
        problem_data['dim'] = d
        # at a given human behaviour
        problem_data['algorithm'] = algorithms[i]
        problem_data['acquisition_function'] = 'UCB'

        plot_regret_batch(problem_data,ax,colors[i],directory)

    fs = 12
    ax.set_ylabel(r"Simple Regret, $r_\tau$",fontsize=fs)
    ax.grid(True,alpha=0.5)
    x_start = problem_data['batch_size']
    max_y = ax.get_ylim()[1]
    min_y = ax.get_ylim()[0]
    ax.plot([x_start,x_start],[min_y,max_y],c='k',ls='--',lw=1,alpha=0.5)

    ax.set_xlabel(r"Batch, $\tau$",fontsize=fs)

    # current xaxis is in units of individual iterations within a batch
    # need to convert to batch number
    iterations = problem_data['max_batches']*problem_data['batch_size']
    x_iteration = [0,iterations/2,iterations]
    batch_iteration = [0,int(problem_data['max_batches']/2),problem_data['max_batches']]
    ax.set_xticks(x_iteration)
    ax.set_xticklabels(batch_iteration)


    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.025), ncol=6,frameon=False)


    #fig.suptitle(r'Regret expectation over 50 functions, $f \sim \mathcal{GP}(\mu \equiv 0, K_M (d,\nu = '+str(l)+'))$, '+str(problem_data['alternatives'])+' alternate choices, $\mathcal{U}(x)=$'+str(aq)+r', $x \in R^'+str(problem_data['dim'])+'$',fontsize=int(fs))
    fig.suptitle('Batch Size = '+str(problem_data['batch_size']))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    plt.savefig('bo/plots/overall_regret_batch_'+str(d)+'.pdf')

# plot_batch(1)
# plot_batch(2)
# plot_batch(5)

def plot_results(folder,name):


    fig,ax = plt.subplots(1,1,figsize=(7.5,3.5))

    files = os.listdir(folder)
    problem_data_list = []
    for i in tqdm(range(len(files))):
        if '.' not in files[i] and '_' not in files[i]:
            results = folder+'/'+files[i] + '/res.json'
            # open json
            with open(results, "r") as f:
                data = json.load(f)
            problem_data_list.append(data['problem_data'])

    # create dataframe from list of dictionaries 
    df = pd.DataFrame(problem_data_list)

    file_names = df['file_name'].values
    obj_list = []
    for file in file_names:
        file = file.split('results/')[1]
        file = folder + '/' +  file 
        data_full = read_json(file+'res.json')
        data = data_full['data']
        obj = [d['objective'] for d in data]
        if len(obj) != problem_data_list[0]['max_iterations']:
            obj += [obj[-1]]*(problem_data_list[0]['max_iterations']-len(obj))

        best_obj = []
        best_obj_val = -1e30
        for o in obj:
            if o > best_obj_val:
                best_obj_val = o
            best_obj.append(best_obj_val)


        obj_list.append(best_obj)

    init = problem_data_list[0]['sample_initial']
    full_it = problem_data_list[0]['max_iterations']


    mean_obj = np.mean(np.array(obj_list),axis=0)
    std_obj = np.std(np.array(obj_list),axis=0)
    # for obj in obj_list:
    #     ax.plot(np.arange(init,len(obj)),obj[init:],c='k',lw=1.5)
    ax.plot(np.arange(init,len(mean_obj)),mean_obj[init:],c='k',lw=1.5,label='Mean')
    ax.fill_between(np.arange(init,len(mean_obj)),mean_obj[init:]-std_obj[init:],mean_obj[init:]+std_obj[init:],alpha=0.1,color='k',lw=0,label='Standard Deviation')


    fs = 12
    ax.set_ylabel(r"Best Objective",fontsize=fs)
    ax.grid(True,alpha=0.5)
    x_start = problem_data_list[0]['sample_initial']
    max_y = ax.get_ylim()[1]
    min_y = ax.get_ylim()[0]
    ax.plot([x_start,x_start],[min_y,max_y],c='k',ls='--',lw=1,alpha=0.5)
    ax.set_xlabel(r"Iteration",fontsize=fs)

    # lines, labels = ax.get_legend_handles_labels()
    # fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.025), ncol=6,frameon=False)
    ax.legend(frameon=False)

    fig.tight_layout()
    # fig.subplots_adjust(bottom=0.2)
    plt.savefig('bo/plots/'+name+'.pdf')

#plot_results('bo/reaction_conditions_results','reaction_conditions')

def plot_regret_llmbo(problem_data,axs,c,directory,function):
    
    files = os.listdir(directory)
    problem_data_list = []
    for i in tqdm(range(len(files))):
        if '.' not in files[i] and function in files[i]:
            results = directory+'/'+files[i] + '/res.json'
            # open json
            with open(results, "r") as f:
                data = json.load(f)
            problem_data_list.append(data['problem_data'])

    # create dataframe from list of dictionaries 
    df = pd.DataFrame(problem_data_list)
    df = df.loc[(df['human_behaviour'] == problem_data['human_behaviour'])]

    file_names = df['file_name'].values
    regret_list = []
    obj_list = []
    f_opt_list = []
    for file in file_names:
        file = file.split('results/')[1]
        file = directory + '/' +  file 
        data_full = read_json(file+'res.json')
        data = data_full['data']
        f_opt = data_full['problem_data']['f_opt']
        try:
            obj = [d['objective'] for d in data]
            f_opt_list.append(f_opt)
            obj_list.append(obj)
        except:
            pass

    init = problem_data['sample_initial']
    full_it = problem_data['max_iterations']

    average_regret_list = []
    regret_list = []
    for obj,i in zip(obj_list,range(len(obj_list))):
        if len(obj) != full_it:
            obj += [obj[-1]]*(full_it-len(obj))

        it = len(obj)
        regret = [f_opt_list[i] - np.max(obj[:t]) for t in range(1,it+1)]
        regret_list.append(regret)
        cumulative_regret = [f_opt_list[i] - np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret = [f_opt_list[i] - (1/t) * np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret_list.append(average_regret)

    n = 100 
    average_regret = np.mean(np.array(average_regret_list),axis=0)[:n]
    average_regret_std = np.std(np.array(average_regret_list),axis=0)[:n]
    regret_list = np.array(regret_list)[:,:n]
    label = problem_data['human_behaviour']

    if label.__class__ != float:
        label = label[0].upper() + label[1:]
    else:
        label = '$p($Best$)=$'+str(label) 
    # label = '$\mathbb{E}[$'+label+'$]$'
    # captialise first letter 

    try:
        x = np.arange(init,len(average_regret))

        axs[1].plot(x,average_regret[init:],c=c,lw=1.5,label=label)
        #axs[1].fill_between(x,average_regret[init:]-average_regret_std[init:],average_regret[init:]+average_regret_std[init:],alpha=0.1,color=c,lw=0)


        ax = axs[0]
        regret_list = np.array(regret_list)
        mean_instantaneous_regret = np.mean(regret_list,axis=0)
        std_instantaneous_regret = np.std(regret_list,axis=0)
        x = np.arange(init,len(mean_instantaneous_regret))
        ax.plot(x,mean_instantaneous_regret[init:],c=c,lw=1.5,label=label)
        lower = mean_instantaneous_regret[init:]-std_instantaneous_regret[init:]
        # cut of less than 0
        lower[lower<0] = 0
        upper = mean_instantaneous_regret[init:]+std_instantaneous_regret[init:]
        #ax.fill_between(x,lower,upper,alpha=0.1,color=c,lw=0)
    except:
        return 
    return 


def plot_llmbo():
    directory = 'bo/benchmark_llmbo_results'
    colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
    # human_behaviours = ['expert','trusting','llmbo',0.33]
    human_behaviours = ['llmbo']

    functions = ['AgNP','AutoAM','Crossed barrel','P3HT','Perovskite']

    for function in functions:
        fig,axs = plt.subplots(1,2,figsize=(8,2.5))
        
        for i in range(len(human_behaviours)):
            # for this problem data
            problem_data = {}
            problem_data["sample_initial"] = 4
            problem_data["gp_ms"] = 8
            problem_data["alternatives"] = 4
            problem_data["NSGA_iters"] = 150
            problem_data["plotting"] = True
            problem_data['max_iterations'] = 60
            problem_data['human_behaviour'] = human_behaviours[i]
            problem_data['acquisition_function'] = 'UCB'

            plot_regret_llmbo(problem_data,axs,colors[i],directory,function)
        fs = 12
        axs[0].set_ylabel(r"Simple Regret, $r_\tau$",fontsize=fs)
        for ax in axs:
            ax.grid(True,alpha=0.5)
            x_start = problem_data['sample_initial']
            max_y = ax.get_ylim()[1]
            min_y = ax.get_ylim()[0]
            ax.plot([x_start,x_start],[min_y,max_y],c='k',ls='--',lw=1,alpha=0.5)
        axs[0].set_xlabel(r"Iterations, $\tau$",fontsize=fs)
        axs[1].set_xlabel(r"Iterations, $\tau$",fontsize=fs)
        axs[1].set_ylabel(r"Average Regret, ${R_\tau}/{\tau}$",fontsize=fs)
        # split function from number at end (2,5 or 10)
        func_name = function
        
        

        # text with white background in upper right of right plot with functon name

        axs[1].text(0.95, 0.95, func_name, horizontalalignment='right',verticalalignment='top', transform=axs[1].transAxes,fontsize=fs,bbox=dict(facecolor='white',edgecolor='none',pad=0.5))
        lines, labels = axs[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.875), ncol=6,frameon=False)

        fig.tight_layout()
        fig.subplots_adjust(top = 0.875,left = 0.125)
        

        #axs[0].set_yscale('log')


        plt.savefig('bo/plots/overall_regret_'+function+'.pdf')
#plot_llmbo()