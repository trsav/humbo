from jax.config import config
from jax import numpy as jnp
from jax import jit, value_and_grad
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
            inputs += [list(d["x"].values())]
            obj += [d["obj"]]
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
    sample = lhs(jnp.array(list(bounds.values())), n, log=False)
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


def lhs(bounds: list, p: int, log):
    d = len(bounds)
    sample = np.zeros((p, len(bounds)))
    for i in range(0, d):
        if log is False:
            sample[:, i] = np.linspace(bounds[i, 0], bounds[i, 1], p)
        else:
            sample[:, i] = np.geomspace(bounds[i, 0], bounds[i, 1], p)
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
        likelihood = gpx.Gaussian(num_datapoints=D.n,obs_noise=0.0)
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
        key=key)
        key,subkey = jax.random.split(key)

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
    #obtain predicted cost 
    f_m, f_v = inference(gp, jnp.array([x]))
    f_v = jnp.sqrt(f_v)
    val = ((-f_v))
    return val[0]
