
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
from function_creation.ce_functions import * 
import multiprocessing as mp


aqs = {'logEI':logEI,'UCB':UCB,'NOISY_EI':noisy_EI,'LETHAM_EI':LETHAM_EI,'EI':EI,'LETHAM_UCB':UCB}


def human():

    problem_data = {}
    problem_data["sample_initial"] = 8
    problem_data["gp_ms"] = 4
    problem_data["alternatives"] = 4
    problem_data["plot"] = False
    problem_data["NSGA_xtol"] = 1e-5
    problem_data["NSGA_ftol"] = 0.01
    problem_data['max_iterations'] = 32
    problem_data['human_behaviour'] = 'trusting'
    problem_data['deterministic_initial'] = 'false'
    problem_data["noisy"] = True
    problem_data['letham_gps'] = 4
    aq = 'LETHAM_UCB'

    problem_data['acquisition_function'] = aq
    problem_data['time_created'] = str(datetime.datetime.now())
    problem_data['include_previous_justification'] = False
    
    # f = BioProcess_Profile(4)
    f = Reactor(4)
    problem_data['noise'] = 0.1 * 0.04
    problem_data['x_names'] = f.x_names
    problem_data['expertise'] = f.expertise
    problem_data['objective_description'] = f.objective_description
    problem_data['function'] = f.name
    problem_data['dim'] = f.dim

    n = 32

    with mp.Pool(8) as pool:
        pool.starmap(llmbo, [(f,aq,problem_data) for i in range(n)])
    # for j in range(16):
    #     llmbo(f,aq,problem_data)

    # f = Reactor(4)
    # problem_data['x_names'] = f.x_names
    # problem_data['expertise'] = f.expertise
    # problem_data['objective_description'] = f.objective_description
    # problem_data['function'] = f.name
    # problem_data['dim'] = f.dim

    # with mp.Pool(4) as pool:
    #     pool.starmap(llmbo, [(f,aq,problem_data) for i in range(n)])


if __name__ == '__main__':

    human()