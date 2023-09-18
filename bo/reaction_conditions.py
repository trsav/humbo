import sys
import os
import jax.numpy as jnp
from jax import vmap
import datetime
import pandas as pd 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.function import SimpleFunction
import matplotlib.pyplot as plt
from algorithm import bo_human
from utils import *
import uuid
from function_creation.create_problem import create_problem
import multiprocessing as mp
import gc 
import resource
import numpy as np
from scipy.integrate import ode




# for this problem data
aq = 'UCB'
problem_data = {}
problem_data["sample_initial"] = 4
problem_data["gp_ms"] = 8
problem_data["alternatives"] = 4
problem_data["NSGA_iters"] = 75
problem_data['max_iterations'] = 75
problem_data['acquisition_function'] = aq

aqs = {'EI':EI,'UCB':UCB}

# for a given function...
file = str(uuid.uuid4())
path = "bo/reaction_conditions_results/" + file + "/"

problem_data['time_created'] = str(datetime.datetime.now())
problem_data['file_name'] = path

def solve_ode(f, c0, t, k):
    c = np.zeros((len(t), len(c0)))
    c[0, :] = c0
    r = ode(f)
    r.set_initial_value(c[0], t[0]).set_f_params(k)

    for k in range(1, len(t)):
        c[k, :] = r.integrate(t[k])
        r.set_initial_value(c[k, :], t[k])
    return c


def kinetic_model(t, z, k):
    c1 = z[0]
    k1 = k[0]
    dzdt = [0 for i in range(len(z))]
    dzdt[0] = -k1 * c1
    dzdt[1] = k1 * c1
    return dzdt


def f(x):

    problem = {"f": kinetic_model, "name": "reaction"}
    problem["tf"] = 4
    problem["n"] = 200
    problem["x0"] = [4, 0]
    problem["param_names"] = ["$k_1$"]
    problem["var_names"] = ["$C_a$", "$C_b$"]
    problem["p"] = [1]


    f = problem["f"]
    tf = problem["tf"]
    n = problem["n"]
    x0 = problem["x0"]
    p_true = problem["p"]

    t = np.linspace(0, tf, n)
    y = solve_ode(f, x0, t, p_true)
    plt.figure()
    for i in range(len(y[0])):
        plt.plot(t, y[:, i])
    plt.show()
    
    return y[0,-1]

f = SimpleFunction(f,[[-5,5],[-5,5]])


bo_human(
    f,
    aqs[aq],
    problem_data
)