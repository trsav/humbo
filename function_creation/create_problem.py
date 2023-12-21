import jax.numpy as jnp
from jax import random
import tqdm as tqdm
import gpjax as gpx
import pickle
from gpjax.kernels import Matern52
import optax as ox
import matplotlib.pyplot as plt


def create_problem(key, l, d):
    n = 300
    x_l = 0.0
    x_u = 1.0

    if d == 1:
        x_b = jnp.linspace(x_l, x_u, n)
    else:
        n = 300 * d
        x_b = jnp.linspace(x_l, x_u, int(n ** (1 / d)))
        x_b = jnp.meshgrid(*[x_b for _ in range(d)])
        x_b = jnp.vstack([x_b[i].flatten() for i in range(d)]).T
        n = x_b.shape[0]
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
    y = (y - jnp.min(y)) / (jnp.max(y) - jnp.min(y))
    if d == 1:
        D = gpx.Dataset(x_b[:, None], y[:, None])
    else:
        D = gpx.Dataset(x_b, y[:, None])

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
        optim=ox.adam(learning_rate=0.2),
        num_iters=750,
        key=random.PRNGKey(0),
    )

    f_dict = {
        "posterior": opt_posterior,
        "D": D,
        "bounds": [[x_l, x_u] for i in range(d)],
        "f_opt": f_opt,
        "dim": d,
    }
    # with open("function_creation/function.pkl", "wb") as f:
    #     pickle.dump(f_dict, f)

    return f_dict


# key = random.PRNGKey(0)
# d = 1 

# fig,axs = plt.subplots(3, 2, figsize=(9, 5),constrained_layout=True)
# for ax,i in zip(axs.ravel(),range(6)):
#     key,subkey = random.split(key)
#     f = Function(create_problem(key,0.04,d))
#     x = jnp.linspace(0,1,200)
#     y = jnp.array([f(x_i) for x_i in x])
#     ax.plot(x,y,c='k',lw=2,alpha=0.75)
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$f(x)$')
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig('function_creation/example_functions.pdf')



# from function import Function 
# fig,axs = plt.subplots(2, 2, figsize=(6, 4.5),constrained_layout=True)
# d = 2 
# key = random.PRNGKey(0)
# for ax,k in zip(axs.ravel(),range(4)):
#     key,subkey = random.split(key)
#     f = Function(create_problem(key,0.04,d))
#     n = 40 
#     x1 = jnp.linspace(0,1,n)
#     x2 = jnp.linspace(0,1,n)
#     X1,X2 = jnp.meshgrid(x1,x2)
#     from tqdm import tqdm 
#     Y = jnp.zeros((n,n))
#     for i in tqdm(range(n)):
#         for j in range(n):
#             Y = Y.at[i,j].set(f(jnp.array([x1[i],x2[j]])))
#     ax.contourf(X1,X2,Y,cmap='Spectral',levels=16)
#     ax.set_xlabel('$x_1$')
#     ax.set_ylabel('$x_2$')
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
# plt.tight_layout()
# plt.savefig('function_creation/example_functions_2D.pdf')