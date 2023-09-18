import sys
import os
import jax.numpy as jnp
from jax import vmap
import datetime
import pandas as pd 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.function import Function
import matplotlib.pyplot as plt
from algorithm import bo
from utils import *
import uuid
from function_creation.create_problem import create_problem
import multiprocessing as mp
import gc 
import resource


def run_behaviour(behaviour_index,aq,d,f_key):

    human_behaviours = ['expert','adversarial','trusting',0.25,0.5,0.75]
    # for this problem data
    problem_data = {}
    problem_data["sample_initial"] = 4
    problem_data["gp_ms"] = 8
    problem_data["alternatives"] = 4
    problem_data["NSGA_iters"] = 75
    problem_data["plotting"] = True
    problem_data['max_iterations'] = 75
    problem_data['lengthscale'] = 0.3
    #problem_data['lengthscale'] = 0.8
    problem_data['dim'] = d
    # at a given human behaviour
    problem_data['human_behaviour'] = human_behaviours[behaviour_index]
    problem_data['acquisition_function'] = aq

    aqs = {'EI':EI,'UCB':UCB}

    # for a given function...
    key = random.PRNGKey(f_key)
    f = Function(create_problem(key,problem_data['lengthscale'],problem_data['dim']))

    file = str(uuid.uuid4())
    # path = "bo/plots/" + file + "/"
    path = "bo/benchmark_results/" + file + "/"

    problem_data['time_created'] = str(datetime.datetime.now())
    problem_data['file_name'] = path
    problem_data['function_key'] = str(f_key)

    bo(
        f,
        aqs[aq],
        problem_data
    )
    return 

if __name__ == '__main__':
    f_keys = pd.read_csv('function_creation/f_keys.csv')['f_keys'].values
    try:
        aq = sys.argv[1]
        d = int(sys.argv[2])
        for b_index in range(6):
            pool = mp.Pool(mp.cpu_count()-2)
            pool.starmap(run_behaviour, [(b_index,aq,d,f_key) for f_key in f_keys])
            pool.close()
            pool.join()
    except:
        aq = 'UCB'
        d = 1
        run_behaviour(0,aq,d,10)

