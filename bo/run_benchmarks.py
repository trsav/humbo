import sys
import os
import jax.numpy as jnp
from jax import vmap
import datetime
import pandas as pd 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.function import *
import matplotlib.pyplot as plt
from llmbo import llmbo
from utils import *
import uuid
from function_creation.create_problem import create_problem
from function_creation.materials_functions import *
import multiprocessing as mp
import gc 
import resource

try:
    arg_check = sys.argv[1]
    local = False
except:
    local = True

aqs = {'EI':EI,'UCB':UCB}

def specific_functions():
    res_path = 'bo/benchmark_results_specific/'
    try:
        os.mkdir(res_path)
    except FileExistsError:
        pass

    aq = 'UCB'
    human_behaviours = ['expert','adversarial','trusting',0.25,0.5,0.75]
    problem_data = {}
    problem_data["sample_initial"] = 4
    problem_data["gp_ms"] = 16
    problem_data["alternatives"] = 4
    problem_data["NSGA_iters"] = 400
    problem_data['max_iterations'] = 80
    problem_data['acquisition_function'] = aq
    problem_data['time_created'] = str(datetime.datetime.now())

    f_store = [Branin(2)]
    for i in [2,5,10]:
        f_store.append(Ackley(i))
        f_store.append(Griewank(i))
        f_store.append(Rastrigin(i))
        f_store.append(Rosenbrock(i))
        f_store.append(Powell(i))

    if local == False:
        repeats = 16
        '''
        # designed to be run as an array job
        # array index 1-256 (for the 16 repeats for 16 functions)
        # b_index as an argument between 0-5 (for the 5 behaviours)
        '''

        array_index = int(sys.argv[1])
        f_key = array_index // repeats
        repeat = array_index % repeats
        b_index = int(sys.argv[2])

    if local == True:
        f_key = 0 
        repeat = 0
        b_index = 0

    f = f_store[f_key]
    file = f.name + '_' + str(uuid.uuid4())
    path = res_path + file + "/"
    problem_data['file_name'] = path
    problem_data['dim'] = f.dim
    problem_data['function'] = f.name
    problem_data['human_behaviour'] = human_behaviours[b_index]

    llmbo(
        f,
        aqs[aq],
        problem_data
    )

#specific_functions()

def rkhs_functions():
    res_path = 'bo/benchmark_results_rkhs/'
    try:
        os.mkdir(res_path)
    except FileExistsError:
        pass

    aq = 'UCB'
    human_behaviours = ['expert','adversarial','trusting',0.25,0.5,0.75]
    problem_data = {}
    problem_data["sample_initial"] = 4
    problem_data["gp_ms"] = 16
    problem_data["alternatives"] = 4
    problem_data["NSGA_iters"] = 400
    problem_data['max_iterations'] = 80
    problem_data['acquisition_function'] = aq
    problem_data['time_created'] = str(datetime.datetime.now())

    f_keys = pd.read_csv('function_creation/f_keys.csv')['f_keys'].values
    f_count = len(f_keys)
    d_store = [1,2,5,10]

    if local == False:
        array_index = int(sys.argv[1])
        f_key = array_index // len(d_store)
        d_ind = array_index % len(d_store)
        d = d_store[d_ind]

        b_index = int(sys.argv[2])
        
    if local == True:
        d = 1
        problem_data['dim'] = d
        b_index = 0 
        f_key = 0


    problem_data['human_behaviour'] = human_behaviours[b_index]
    key = random.PRNGKey(f_keys[f_key])
    f = Function(create_problem(key,0.04,problem_data['dim']))

    problem_data['dim'] = d
    file = str(uuid.uuid4())
    path = res_path + file + "/"
    problem_data['file_name'] = path
    problem_data['function'] = file

    llmbo(
        f,
        aqs[aq],
        problem_data
    )

# rkhs_functions()


def real_functions():
    res_path = 'bo/benchmark_results_real/'
    try:
        os.mkdir(res_path)
    except FileExistsError:
        pass

    def create_P3HT():
        f = P3HT(8)
        return  
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

    human_behaviours = ['llmbo',0.33,'expert','trusting']

    aq = 'UCB'
    problem_data = {}
    problem_data["sample_initial"] = 4
    problem_data["gp_ms"] = 16
    problem_data["alternatives"] = 4
    problem_data["NSGA_iters"] = 400
    problem_data['max_iterations'] = 80
    problem_data['gpt'] = 3.5
    problem_data['acquisition_function'] = aq
    problem_data['time_created'] = str(datetime.datetime.now())

    if local == False:
        array_index = int(sys.argv[1])
        f_index = array_index // repeats
        repeat = array_index % repeats
        b_index = int(sys.argv[2]) # per job script 
    if local == True:
        b_index = 1
        f_index = 1
        repeat = 0 

    problem_data['human_behaviour'] = human_behaviours[b_index]
    f = f_list[f_index]()
    problem_data['repeat'] = repeat
    problem_data['x_names'] = f.x_names
    problem_data['expertise'] = f.expertise
    problem_data['objective_description'] = f.objective_description
    problem_data['function'] = f.name
    problem_data['dim'] = f.dim
    problem_data['human_behaviour'] = human_behaviours[b_index]
    problem_data['include_previous_justification'] = True 
    file = f.name + '_' + str(uuid.uuid4())
    path = res_path + file + "/"
    problem_data['file_name'] = path
        
    llmbo(
        f,
        aqs[aq],
        problem_data
    )

real_functions()

