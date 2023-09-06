import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from bo.utils import *
from bo.main import *
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
        self.bounds = {"x": self.gp["bounds"]}
        self.f_opt = self.gp["f_opt"]

    def __call__(self, theta):
        x = theta["x"]
        latent_dist = self.opt_posterior.predict(jnp.array([[x]]), train_data=self.D)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)
        f = predictive_dist.mean()
        return f.item()

    def eval_vector(self, x):
        latent_dist = self.opt_posterior.predict(jnp.array([x]).T, train_data=self.D)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)
        return predictive_dist.mean()
