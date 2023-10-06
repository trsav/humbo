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
from function_creation.materials_functions import *
import multiprocessing as mp
import gc 
import resource


def run_behaviour(f,res_path,problem_data):

    aqs = {'EI':EI,'UCB':UCB}
    file = f.name + str(uuid.uuid4())
    path = res_path + file + "/"
    problem_data['time_created'] = str(datetime.datetime.now())
    problem_data['file_name'] = path

    llmbo(
        f,
        aqs[problem_data['acquisition_function']],
        problem_data
    )
    return 

f = P3HT(1)

if __name__ == '__main__':


    res_path = 'bo/benchmark_llmbo_results/'

    
    problem_data = {}
    problem_data["sample_initial"] = 8
    problem_data["gp_ms"] = 8
    problem_data["alternatives"] = 4
    problem_data["NSGA_iters"] = 40
    problem_data["plotting"] = False
    problem_data['max_iterations'] = 100
    problem_data['acquisition_function'] = 'UCB'
    problem_data['x_names'] = f.x_names
    problem_data['expertise'] = f.expertise
    problem_data['objective_description'] = f.objective_description
    problem_data['function'] = f.name
    problem_data['dim'] = f.dim
    problem_data['human_behaviour'] = 'llmbo'
    problem_data['include_previous_justification'] = True 
    problem_data['gpt'] = 3.5
    run_behaviour(f,res_path,problem_data)

# todo benchmark previous justifications (yes/no)
# todo benchmark different problems and what complexity the benefits reduce



