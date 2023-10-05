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


def run_behaviour(f,res_path,problem_data):

    aqs = {'EI':EI,'UCB':UCB}
    file = f.name
    path = res_path + file + "/"
    problem_data['time_created'] = str(datetime.datetime.now())
    problem_data['file_name'] = path

    llmbo(
        f,
        aqs[problem_data['acquisition_function']],
        problem_data
    )
    return 

d = 5
f = Griewank(d)
repeats = 8
x_names = ["x"+str(i) for i in range(d)]
expertise = 'Optimising Benchmark Functions'
objective_desc = "Maximise the negative "+f.name+" function."

human_behaviours = ['llmbo','expert','trusting',0.25]
if __name__ == '__main__':
    try:
        repeat = int(sys.argv[1])
        aq = 'UCB'
        res_path = 'bo/benchmark_llmbo_results/'
        for behav in human_behaviours:
            run_behaviour(behav,aq,f,res_path)
    except:

        res_path = 'bo/benchmark_llmbo_results/'
        problem_data_init = {}
        problem_data_init['x_names'] = x_names
        problem_data_init['expertise'] = expertise
        problem_data_init['objective_description'] = objective_desc
        problem_data_init['function'] = f.name
        problem_data_init["sample_initial"] = 8
        problem_data_init["gp_ms"] = 8
        problem_data_init["alternatives"] = 4
        problem_data_init["NSGA_iters"] = 200
        problem_data_init["plotting"] = False
        problem_data_init['max_iterations'] = 100
        problem_data_init['dim'] = f.dim
        problem_data_init['human_behaviour'] = 'llmbo'
        problem_data_init['acquisition_function'] = 'UCB'
        problem_data_init['include_previous_justification'] = True 
        problem_data_init['gpt'] = 3.5
        run_behaviour(f,res_path,problem_data_init)

# benchmark previous justifications (yes/no)
# benchmark LLM (gpt3.5/gpt4)
# benchmark different problems and what complexity the benefits reduce

