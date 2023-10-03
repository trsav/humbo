import sys
import os
import jax.numpy as jnp
from jax import vmap
import datetime
import pandas as pd 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.function import *
import matplotlib.pyplot as plt
from algorithm import bo
from utils import *
import uuid
from function_creation.create_problem import create_problem
import multiprocessing as mp
import gc 
import resource


def run_behaviour(behaviour_index,aq,f,res_path):

    
    human_behaviours = ['expert','adversarial','trusting',0.25,0.5,0.75]
    # for this problem data
    problem_data = {}
    problem_data['function'] = f.name
    problem_data["sample_initial"] = 4
    problem_data["gp_ms"] = 8
    problem_data["alternatives"] = 4
    problem_data["NSGA_iters"] = 150
    problem_data["plotting"] = False
    problem_data['max_iterations'] = 100
    #problem_data['lengthscale'] = 0.8
    problem_data['dim'] = d
    # at a given human behaviour
    problem_data['human_behaviour'] = human_behaviours[behaviour_index]
    problem_data['acquisition_function'] = aq

    aqs = {'EI':EI,'UCB':UCB}

    file = f.name + '_' + str(uuid.uuid4())
    path = res_path + file + "/"

    problem_data['time_created'] = str(datetime.datetime.now())
    problem_data['file_name'] = path
    # problem_data['function_key'] = str(f_key)

    bo(
        f,
        aqs[aq],
        problem_data
    )
    return 


f_store = [Branin(2)]
for i in [2,5,10]:
    f_store.append(Ackley(i))
    f_store.append(Griewank(i))
    f_store.append(Rastrigin(i))
    f_store.append(Rosenbrock(i))
    f_store.append(Powell(i))

repeats = 8


if __name__ == '__main__':
    try:
        f_index = int(sys.argv[1])
        # split f_index into f_key and repeat
        f_key = f_index // repeats
        repeat = f_index % repeats
        aq = 'UCB'
        res_path = 'bo/benchmark_results_specific/'
        for b_index in range(6):
            run_behaviour(b_index,aq,f_store[f_key],res_path)
    except:
        aq = 'UCB'
        d = 2
        res_path = 'bo/plots/'
        f_key = f_store[2]
        run_behaviour(2,aq,f_key,res_path)

