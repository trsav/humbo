import jax.numpy as jnp
from jax import random
import tqdm as tqdm
import gpjax as gpx
import pickle
from gpjax.kernels import Matern52
import optax as ox


def create_problem(key,l):
    n = 200
    x_l = 0.0
    x_u = 10.0
    x_b = jnp.linspace(x_l, x_u, n)
    # X_n = jnp.meshgrid(x_b,x_b)
    # now do vectorised version
    sigma = 1
    k = Matern52(variance=sigma, lengthscale=l)
    gram_matrix = k.gram(x_b).matrix

    y = random.multivariate_normal(key, jnp.zeros(n), gram_matrix)
    # w = jnp.linspace(1, 1.5, int(n / 2))
    # w = jnp.concatenate((w, w[::-1]))
    # y *= w
    # y = y[:, None]
    D = gpx.Dataset(x_b[:, None], y[:,None])

    f_opt = jnp.max(y)
    kernel = gpx.kernels.Matern52()
    meanf = gpx.mean_functions.Constant()
    prior = gpx.Prior(mean_function=meanf, kernel=kernel)
    posterior = prior * gpx.likelihoods.Gaussian(num_datapoints=D.n, obs_noise=0.0)
    negative_mll = gpx.objectives.ConjugateMLL(negative=True)
    negative_mll(posterior, train_data=D)

    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=negative_mll,
        train_data=D,
        optim=ox.adam(learning_rate=0.1),
        num_iters=100,
        safe=True,
        key=random.PRNGKey(0),
    )

    f_dict = {"posterior": opt_posterior, "D": D, "bounds": [x_l, x_u], "f_opt": f_opt}
    with open("function_creation/function.pkl", "wb") as f:
        pickle.dump(f_dict, f)

    return f_dict
