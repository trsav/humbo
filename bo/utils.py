from jax.config import config
import blackjax
from jax import numpy as jnp
from jax import jit, value_and_grad, vmap
import matplotlib
import jax.random as random
from pymoo.core.problem import ElementwiseProblem, Problem

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


def sample_bounds(bounds, n):
    sample = lhs(jnp.array(list(bounds.values())), n)
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
        s = jax.random.shuffle(random.PRNGKey(0), s)
        sample.append(s)
    sample = jnp.array(sample).T

    return sample


class HumanInterface:
    def __init__(self, name, var_names):
        self.name = name
        self.var_names = var_names

    def describe(self, x):
        return "".join(
            [self.var_names[i] + ": " + str(x[i]) + "\n" for i in range(len(x))]
        )



def train_gp(inputs, outputs, ms):
    # creating a set of initial GP hyper parameters (log-spaced)
    p_num = len(inputs[0, :])

    init_params = np.zeros((ms, p_num))
    for i in range(0, p_num):
        init_params[:, i] = np.geomspace(0.01, 1, ms)
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
            num_iters=750,
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


def aq(x, args):
    gp, f_best = args
    m, K = inference(gp, jnp.array([x]))
    sigma = jnp.sqrt(K)
    diff = m - f_best
    p_y = tfd.Normal(loc=m, scale=sigma)
    Z = diff / sigma
    return -(diff * p_y.cdf(Z) + sigma * p_y.prob(Z))[0]
