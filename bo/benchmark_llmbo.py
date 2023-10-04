import sys
import os
import jax.numpy as jnp
from jax import vmap
import datetime
import pandas as pd 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.function import *
import matplotlib.pyplot as plt
from llm_bo import llmbo
from utils import *
import uuid
from function_creation.create_problem import create_problem
import multiprocessing as mp
import gc 
import resource


def run_behaviour(behaviour,aq,f,res_path,problem_data):

    problem_data['function'] = f.name
    problem_data["sample_initial"] = 8
    problem_data["gp_ms"] = 8
    problem_data["alternatives"] = 4
    problem_data["NSGA_iters"] = 200
    problem_data["plotting"] = False
    problem_data['max_iterations'] = 100
    #problem_data['lengthscale'] = 0.8
    problem_data['dim'] = f.dim
    # at a given human behaviour
    problem_data['human_behaviour'] = behaviour
    problem_data['acquisition_function'] = aq

    aqs = {'EI':EI,'UCB':UCB}

    file = f.name

    path = res_path + file + "/"

    problem_data['time_created'] = str(datetime.datetime.now())
    problem_data['file_name'] = path
    # problem_data['function_key'] = str(f_key)

    llmbo(
        f,
        aqs[aq],
        problem_data
    )
    return 


f = Rosenbrock(5)
repeats = 8
x_names = ["x1", "x2", "x3", "x4", "x5"]
expertise = 'Optimising Benchmark Functions'
objective_desc = "Maximise the negative Rosenbrock function."

human_behaviours = ['llmbo','expert','trusting',0.25]
if __name__ == '__main__':
    try:
        repeat = int(sys.argv[1])
        aq = 'UCB'
        res_path = 'bo/benchmark_llmbo_results/'
        for behav in human_behaviours:
            run_behaviour(behav,aq,f,res_path)
    except:
        aq = 'UCB'
        res_path = 'bo/benchmark_llmbo_results/'
        problem_data_init = {}
        problem_data_init['x_names'] = x_names
        problem_data_init['expertise'] = expertise
        problem_data_init['objective_description'] = objective_desc
        run_behaviour('llmbo',aq,f,res_path,problem_data_init)

