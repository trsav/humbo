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

def create_P3HT():
    f = P3HT(8)
    return f 

def create_AgNP():
    f = AgNP(8)
    return f

def create_Perovskite():
    f = Perovskite(8)
    return f

def create_AutoAM():
    f = AutoAM(8)
    return f

def create_CrossedBarrel():
    f = CrossedBarrel(1)
    return f

f_list = [create_P3HT,create_AgNP,create_Perovskite,create_AutoAM,create_CrossedBarrel]



repeats = 8 
f_count = 5 

if __name__ == '__main__':

    human_behaviours = [0.33,'expert','trusting']

    res_path = 'bo/benchmark_llmbo_results/'

    try:
        array_index = int(sys.argv[1])
        f_key = array_index // repeats
        repeat = array_index % repeats
        
        b_index = int(sys.argv[2]) # per job script 
    except:
        f_key = 1
        b_index = 3
        repeat = 0 
        res_path = 'bo/plots/'

    f = f_list[f_key]()
    #f = RosenbrockLLM(5)
    
    problem_data = {}
    problem_data["sample_initial"] = 16
    problem_data['repeat'] = repeat
    problem_data["gp_ms"] = 16
    problem_data["alternatives"] = 3
    problem_data["NSGA_iters"] = 1000
    problem_data['max_iterations'] = 50
    problem_data['acquisition_function'] = 'UCB'
    problem_data['x_names'] = f.x_names
    problem_data['expertise'] = f.expertise
    problem_data['objective_description'] = f.objective_description
    problem_data['function'] = f.name
    problem_data['dim'] = f.dim
    problem_data['human_behaviour'] = human_behaviours[b_index]
    problem_data['include_previous_justification'] = True 
    problem_data['gpt'] = 3.5

    run_behaviour(f,res_path,problem_data)

    if human_behaviours[b_index] == 'llmbo':
        problem_data['human_behaviour'] = 'llmbo_no_prev_justification'
        problem_data['include_previous_justification'] = False
        run_behaviour(f,res_path,problem_data)




