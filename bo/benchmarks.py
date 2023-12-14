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
from function_creation.ce_functions import *
import multiprocessing as mp
import gc 
import resource
import argparse

aqs = {'EI':EI,'UCB':UCB}

def specific_functions(array_index,b_index):
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
    problem_data["NSGA_iters"] = 1000
    problem_data['max_iterations'] = 60
    problem_data['acquisition_function'] = aq
    problem_data['time_created'] = str(datetime.datetime.now())

    f_store = [Branin(2)]
    for i in [2,5,10]:
        f_store.append(Ackley(i))
        f_store.append(Griewank(i))
        f_store.append(Rastrigin(i))
        f_store.append(Rosenbrock(i))
        f_store.append(Powell(i))

    repeats = 32
    f_key = array_index // repeats
    repeat = array_index % repeats


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

def rkhs_functions(array_index, b_index):
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
    problem_data["NSGA_iters"] = 1000
    problem_data['deterministic_initial'] = 'true'
    problem_data['max_iterations'] = 60
    problem_data['acquisition_function'] = aq
    problem_data['time_created'] = str(datetime.datetime.now())

    f_keys = pd.read_csv('function_creation/f_keys.csv')['f_keys'].values
    f_count = len(f_keys)
    d_store = [1,2,5]

    f_key = array_index // len(d_store)
    d_ind = array_index % len(d_store)
    d = d_store[d_ind]
    problem_data['dim'] = d

    problem_data['human_behaviour'] = human_behaviours[b_index]
    key = random.PRNGKey(f_keys[f_key])
    f = Function(create_problem(key,0.04,problem_data['dim']))

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

def real_functions(array_index,b_index):
    res_path = 'bo/benchmark_results_real/'
    try:
        os.mkdir(res_path)
    except FileExistsError:
        pass
    
    def create_SelfOpt():
        f = SelfOpt(4)
        return f
    def create_AgNP():
        f = AgNP(4)
        return f
    def create_Perovskite():
        f = Perovskite(4)
        return f
    def create_CrossedBarrel():
        f = CrossedBarrel(1)
        return f
    
    def create_Reactor():
        f = Reactor(1)
        return f
    
    f_list = [create_Reactor,create_SelfOpt,create_AgNP,create_Perovskite,create_CrossedBarrel]

    repeats = 16
    f_key = array_index // repeats
    repeat = array_index % repeats
    f_key = 0 
    print(repeats,f_key)

    # human_behaviours = ['llmbo',0.25,'expert','trusting']
    human_behaviours = ['expert','trusting']

    aq = 'EI'
    problem_data = {}
    problem_data["sample_initial"] = 8
    problem_data["gp_ms"] = 8
    problem_data["alternatives"] = 3
    problem_data["NSGA_iters"] = 1000
    problem_data['max_iterations'] = 50
    problem_data['acquisition_function'] = aq
    problem_data['time_created'] = str(datetime.datetime.now())
    
    problem_data['human_behaviour'] = human_behaviours[b_index]
    f = f_list[f_key]()
    problem_data['repeat'] = repeat
    problem_data['x_names'] = f.x_names
    problem_data['expertise'] = f.expertise
    problem_data['objective_description'] = f.objective_description
    problem_data['function'] = f.name
    problem_data['dim'] = f.dim
    #problem_data['llm_location'] = 'remote'
    # problem_data['llm_location'] = "llama.cpp/models/13B/ggml-model-q8_0.gguf"
    problem_data['llm_location'] = "llama.cpp/models/zephyr-7b-alpha.Q4_K_M.gguf"

    problem_data['human_behaviour'] = human_behaviours[b_index]
    problem_data['include_previous_justification'] = False
    file = f.name + '_' + str(uuid.uuid4())
    path = res_path + file + "/"
    problem_data['file_name'] = path
        
    llmbo(
        f,
        aqs[aq],
        problem_data
    )

    # problem_data['include_previous_justification'] = True
    # file = f.name + '_' + str(uuid.uuid4())
    # path = res_path + file + "/"
    # problem_data['file_name'] = path
        
    # llmbo(
    #     f,
    #     aqs[aq],
    #     problem_data
    # )

#real_functions()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run specific function.")
    parser.add_argument('--function', type=str, required=True, help="Function name to run.")
    parser.add_argument('--array_index', type=str, required=False, help="First index.")
    parser.add_argument('--behaviour_index', type=str, required=False, help="Behaviour index.")

    try:
        args = parser.parse_args()
    except:
        # rkhs_functions(0,0)
        for b in range(4):
            real_functions(2,0)
        # specific_functions(0,2)
    
    a_i = int(args.array_index)
    b_i = int(args.behaviour_index)

    if args.function == "real_functions":
        real_functions(a_i,b_i)
    elif args.function == "rkhs_functions":
        rkhs_functions(a_i,b_i)
    elif args.function == "specific_functions":
        specific_functions(a_i,b_i)

