from jax.config import config
import blackjax
from jax import numpy as jnp
from jax import jit, value_and_grad, vmap
import matplotlib

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
from scipy.optimize import minimize
from tensorflow_probability.substrates import jax as tfp

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
            inputs += [list(d["inputs"].values())]
            obj += [d["objective"]]
            cost += [d["cost"]]

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


def sample_bounds(bounds, n, key):
    sample = lhs(jnp.array(list(bounds.values())), n, key, random=True)
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
    return data


def lhs(bounds: list, p: int, key,random):
    d = len(bounds)
    sample = np.zeros((p, len(bounds)))
    for i in range(0, d):
        if random is False:
            sample[:, i] = np.linspace(bounds[i, 0], bounds[i, 1], p)
        else:
            sample[:, i] = jnp.random.uniform(xmin=bounds[i, 0], xmax=bounds[i, 1], size=(p),key=key)
        if random is False:
            rnd.shuffle(sample[:, i])
    return sample


def train_gp(inputs, outputs, ms):
    key = jax.random.PRNGKey(0)
    # creating a set of initial GP hyper parameters (log-spaced)
    init_params = lhs(
        np.array([[0.1, 10] for i in range(len(inputs[0, :]))]), ms, log=True
    )
    # defining dataset
    D = gpx.Dataset(X=inputs, y=outputs)
    # for each intital list of hyperparameters
    best_nll = 1e30
    for p in init_params:
        kernel = gpx.kernels.Matern52()
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
            num_iters=500,
            safe=True,
            key=key,
        )
        key, subkey = jax.random.split(key)

        nll = float(history[-1])
        # if this is the best, then store this
        if nll < best_nll:
            best_nll = nll
            best_posterior = opt_posterior
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


def exp_design_hf(x, gp):
    # obtain predicted cost
    f_m, f_v = inference(gp, jnp.array([x]))
    f_v = jnp.sqrt(f_v)
    val = -f_v
    return val[0]


@jit
def calculate_hf_entropy_sample(x, x_s, y_s, l_s, gp):
    D = gp["D"] + gpx.Dataset(X=x, y=jnp.array([y_s]))
    kernel = gpx.kernels.Matern52(lengthscale=l_s)
    meanf = gpx.mean_functions.Constant()
    prior = gpx.Prior(mean_function=meanf, kernel=kernel)
    likelihood = gpx.Gaussian(num_datapoints=D.n, obs_noise=0.0)
    posterior = prior * likelihood
    gp_s = build_gp_dict(posterior, D)
    m_s, v_s = inference(gp_s, jnp.array([x_s]))
    return v_s[0]


def exp_design_mf(x, gp, c_gp, z_high, x_bounds):
    x = jnp.array([x])

    n = 500
    # sampling from x space
    x_s_list = lhs(x_bounds, n, log=False)
    # appending highest fidelities
    x_s_list = jnp.concatenate((x_s_list, jnp.ones((n, len(z_high))) * z_high), axis=1)

    m, v = inference(gp, x)
    std = jnp.sqrt(v)
    # defining prior distribution of output values
    y_d = tfd.Normal(loc=m[0], scale=std[0])

    # defining prior distribution of lengthscales
    gp_l = jnp.array([gp["posterior"].prior.kernel.lengthscale])
    l_d = tfd.MultivariateNormalDiag(loc=gp_l, scale_diag=gp_l / 4)

    key = jax.random.PRNGKey(0)
    y_key, l_key = jax.random.split(key)
    y_s_list = y_d.sample(n, seed=y_key).reshape(-1, 1)
    l_s_list = l_d.sample(n, seed=l_key).reshape(-1, 1)

    batched_iteration = vmap(calculate_hf_entropy_sample, in_axes=(None, 0, 0, 0, None))
    approx_entropy = jnp.sum(batched_iteration(x, x_s_list, y_s_list, l_s_list, gp))

    c_m, c_v = inference(c_gp, x)

    # minimize
    return -(-approx_entropy / c_m[0])
