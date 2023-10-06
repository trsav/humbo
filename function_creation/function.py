import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from bo.utils import *
from bo.algorithm import *
import uuid
import pickle


class Function:
    def __init__(self, info):
        if info.__class__ == dict:
            self.gp = info
        elif info.__class__ == str:
            self.path = info
            self.gp = pickle.load(open(self.path, "rb"))
        self.opt_posterior = self.gp["posterior"]
        self.D = self.gp["D"]
        self.dim = self.gp["dim"]
        self.bounds = self.gp["bounds"]
        self.f_opt = self.gp["f_opt"].item()

    def __call__(self, x):
        # unnormalise
        latent_dist = self.opt_posterior.predict(jnp.array([x]), train_data=self.D)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)
        f = predictive_dist.mean()
        return f.item()

    def eval_vector(self, x):
        latent_dist = self.opt_posterior.predict(jnp.array([x]).T, train_data=self.D)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)
        return predictive_dist.mean()


class SimpleFunction:
    def __init__(self, f, bounds):
        self.bounds = bounds
        self.f = f
        self.dim = len(self.bounds)

    def __call__(self, x, ax):
        return self.f(x, ax)


class Branin:
    def __init__(self, d):
        self.name = "Branin"
        # self.bounds = jnp.array([[-5, 10], [0, 15]])
        self.bounds = jnp.array([[-0.5, 1], [0, 1]])
        self.f_opt = -0.397887
        self.dim = 2

    def __call__(self, x):
        x1 = x[0] * 10
        x2 = x[1] * 15 
        a = 1
        b = 5.1 / (4 * jnp.pi**2)
        c = 5 / jnp.pi
        r = 6
        s = 10
        t = 1 / (8 * jnp.pi)
        f = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * jnp.cos(x1) + s
        return -f.item()


class Ackley:
    def __init__(self, d):
        self.name = "Ackley" + str(d)
        # self.bounds = jnp.array([[-32.768, 32.768]] * d)
        self.bounds = jnp.array([[-1, 1]] * d)
        self.f_opt = 0
        self.dim = d

    def __call__(self, x):
        d = len(x)
        x = [x[i] * 32.768 for i in range(d)]
        x = jnp.array(x)
        a = 20
        b = 0.2
        c = 2 * jnp.pi
        f = (
            -a * jnp.exp(-b * jnp.sqrt(1 / d * jnp.sum(x**2)))
            - jnp.exp(1 / d * jnp.sum(jnp.cos(c * x)))
            + a
            + jnp.exp(1)
        )
        return -f.item()


class Griewank:
    def __init__(self, d):
        self.name = "Griewank" + str(d)
        # self.bounds = jnp.array([[-600, 600]] * d)
        self.bounds = jnp.array([[-1, 1]] * d)
        self.f_opt = 0
        self.dim = d

    def __call__(self, x):
        d = len(x)
        x = [x[i] * 600 for i in range(d)]
        x = jnp.array(x)

        a = 1 / 4000 * jnp.sum(x**2)
        b = jnp.prod(jnp.cos(x / jnp.sqrt(jnp.arange(1, d + 1))))
        f = a - b + 1
        return -f.item()


class Rastrigin:
    def __init__(self, d):
        self.name = "Rastrigin" + str(d)
        # self.bounds = jnp.array([[-5.12, 5.12]] * d)
        self.bounds = jnp.array([[-1, 1]] * d)

        self.f_opt = 0
        self.dim = d

    def __call__(self, x):
        d = len(x)
        x = [x[i] * 5.12 for i in range(d)]
        x = jnp.array(x)
        a = 10 * d
        b = jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))
        f = a + b
        return -f.item()


class Rosenbrock:
    def __init__(self, d):
        self.name = "Rosenbrock" + str(d)
        # self.bounds = jnp.array([[-5, 10]] * d)
        self.bounds = jnp.array([[-0.5, 1]] * d)
        self.f_opt = 0
        self.dim = d

    def __call__(self, x):
        d = len(x)
        x = [x[i] * 10 for i in range(d)]
        x = jnp.array(x)
        f = jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)
        return -f.item()


class Powell:
    def __init__(self, d):
        self.name = "Powell" + str(d)
        # self.bounds = jnp.array([[-4, 5]] * d)
        self.bounds = jnp.array([[-0.4, 0.5]] * d)
        self.f_opt = 0
        self.dim = d

    def __call__(self, x):
        d = len(x)
        x = [x[i] * 10 for i in range(d)]
        x = jnp.array(x)
        f = jnp.sum(
            (x[0::4] - 10 * x[1::4]) ** 2
            + 5 * (x[2::4] - x[3::4]) ** 2
            + (x[1::4] - 2 * x[2::4]) ** 4
            + 10 * (x[0::4] - x[3::4]) ** 4
        )
        return -f.item()
