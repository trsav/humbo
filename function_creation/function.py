import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from bo.utils import *
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
    '''
    LaTeX: f(\mathbf{x}) = a (x_2 - b x_1^2 + c x_1 - r)^2 + s (1 - t) \cos(x_1) + s 
    where a = 1, b = 5.1 / (4 \pi^2), c = 5 / \pi, r = 6, s = 10, t = 1 / (8 \pi)
    '''
    def __init__(self, d):
        self.name = "Branin"
        self.bounds = jnp.array([[-5, 10], [0, 15]])
        self.f_opt = -0.397887
        self.f_max = 300 # approximately 
        self.dim = 2

    def __call__(self, x):
        x1, x2 = x
        a = 1
        b = 5.1 / (4 * jnp.pi**2)
        c = 5 / jnp.pi
        r = 6
        s = 10
        t = 1 / (8 * jnp.pi)
        f = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * jnp.cos(x1) + s
        return -f.item()


class Ackley:
    '''
    LaTeX: f(\mathbf{x}) = -a \exp \left( -b \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2} \right) - \exp \left( \frac{1}{d} \sum_{i=1}^d \cos(c x_i) \right) + a + \exp(1)
    where a = 20, b = 0.2, c = 2 \pi
    '''
    def __init__(self, d):
        self.name = "Ackley" + str(d)
        self.bounds = jnp.array([[-32.768, 32.768]] * d)
        self.f_opt = 0
        self.f_max = 22 # approximately
        self.dim = d

    def __call__(self, x):
        d = len(x)
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
    '''
    LaTeX: f(\mathbf{x}) = \frac{1}{4000} \sum_{i=1}^d x_i^2 - \prod_{i=1}^d \cos \left( \frac{x_i}{\sqrt{i}} \right) + 1
    '''
    
    
    def __init__(self, d):
        self.name = "Griewank" + str(d)
        self.bounds = jnp.array([[-600, 600]] * d)
        self.f_opt = 0
        self.f_max = 150 # approximately
        self.dim = d

    def __call__(self, x):
        d = len(x)
        x = jnp.array(x)

        a = 1 / 4000 * jnp.sum(x**2)
        b = jnp.prod(jnp.cos(x / jnp.sqrt(jnp.arange(1, d + 1))))
        f = a - b + 1
        return -f.item()


class Rastrigin:
    '''
    LaTeX: f(\mathbf{x}) = 10d + \sum_{i=1}^d \left[ x_i^2 - 10 \cos(2 \pi x_i) \right] 
    '''
    def __init__(self, d):
        self.name = "Rastrigin" + str(d)
        self.bounds = jnp.array([[-5.12, 5.12]] * d)

        self.f_opt = 0
        self.f_max = 80 # approximately
        self.dim = d

    def __call__(self, x):
        d = len(x)
        x = jnp.array(x)
        a = 10 * d
        b = jnp.sum(x**2 - 10 * jnp.cos(2 * jnp.pi * x))
        f = a + b
        return -f.item()


class Rosenbrock:
    '''
    LaTeX: f(\mathbf{x}) = \sum_{i=1}^{d-1} \left[ 100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2 \right]
    '''
    def __init__(self, d):
        self.name = "Rosenbrock" + str(d)
        self.bounds = jnp.array([[-5, 10]] * d)
        self.f_opt = 0
        self.f_max = 15 # approximately
        self.dim = d

    def __call__(self, x):
        d = len(x)
        x = jnp.array(x)
        f = jnp.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)
        return -f.item()
    
class Levi:
    '''
    LaTeX: f(\mathbf{x}) = \sin^2(\pi w_1) + \sum_{i=1}^{d-1} \left[ (w_i - 1)^2 (1 + 10 \sin^2(\pi w_i + 1)) + (w_d - 1)^2 (1 + \sin^2(2 \pi w_d)) \right]
    where w_i = 1 + \frac{x_i - 1}{4}
    '''

    def __init__(self,d):
        self.name = "Levi" + str(d)
        self.bounds = jnp.array([[-10, 10]] * d)
        self.f_opt = 0
        self.f_max = 80
        self.dim = d
    
    def __call__(self,x):
        d = len(x)
        x = jnp.array(x)
        w = 1 + (x - 1) / 4
        f = jnp.sin(jnp.pi * w[0]) ** 2 + jnp.sum((w[:-1] - 1) ** 2 * (1 + 10 * jnp.sin(jnp.pi * w[:-1] + 1) ** 2)) + (w[-1] - 1) ** 2 * (1 + jnp.sin(2 * jnp.pi * w[-1]) ** 2)
        return -f.item()

class Schwefel:
    def __init__(self,d):
        self.name = "Schewefel" + str(d)
        self.bounds = jnp.array([[-500, 500]] * d)
        self.f_opt = 0
        self.f_max = 1600
        self.dim = d

    def __call__(self,x):
        d = len(x)
        x = jnp.array(x)
        f = 418.9829 * d - jnp.sum(x * jnp.sin(jnp.sqrt(jnp.abs(x))))
        return -f.item()
  

class StyblinskiTang:
    '''
    LaTeX: f(\mathbf{x}) = \frac{1}{2} \sum_{i=1}^d \left[ x_i^4 - 16 x_i^2 + 5 x_i \right] 
    '''
    def __init__(self, d):
        self.name = "StyblinskiTang" + str(d)
        self.bounds = jnp.array([[-5, 5]] * d)
        self.f_opt = 0
        self.f_max = 250
        self.dim = d

    def __call__(self, x):
        d = len(x)
        x = jnp.array(x)
        f = 0.5 * jnp.sum(x**4 - 16 * x**2 + 5 * x)
        return -f.item()
