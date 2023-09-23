import sys
import os
import jax.numpy as jnp
from jax import vmap
import datetime
import pandas as pd 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.function import Function
import matplotlib.pyplot as plt
from algorithm import bo_batch
from utils import *
import uuid
from function_creation.create_problem import create_problem
import multiprocessing as mp
import gc 
import resource


def run_behaviour(algo,aq,d,f_key,res_path):

    # for this problem data
    problem_data = {}
    problem_data['algorithm'] = algo
    problem_data["batch_size"] = 8
    problem_data["gp_ms"] = 8
    problem_data["NSGA_iters"] = 75
    problem_data["plotting"] = False
    problem_data['max_batches'] = 50
    problem_data['lengthscale'] = 0.3
    #problem_data['lengthscale'] = 0.8
    problem_data['dim'] = d
    # at a given human behaviour
    problem_data['acquisition_function'] = aq

    aqs = {'EI':EI,'UCB':UCB}

    # for a given function...
    key = random.PRNGKey(f_key)
    f = Function(create_problem(key,problem_data['lengthscale'],problem_data['dim']))

    file = str(uuid.uuid4())
    # path = "bo/plots/" + file + "/"
    #file = 'batch'
    path = res_path + file + "/"

    problem_data['time_created'] = str(datetime.datetime.now())
    problem_data['file_name'] = path
    problem_data['function_key'] = str(f_key)

    bo_batch(
        f,
        aqs[aq],
        problem_data
    )
    return 

if __name__ == '__main__':
    f_keys = pd.read_csv('function_creation/f_keys.csv')['f_keys'].values
    try:
        aq = sys.argv[1]
        f_index = int(sys.argv[2])
        res_path = 'bo/batch_benchmark_results/'
        for d in [1,2,5]:
            for algo in ['random']:
                run_behaviour(algo,aq,d,f_keys[f_index],res_path)
    except:
        aq = 'UCB'
        d = 5
        res_path = 'bo/batch_benchmark_results/'
        algo = 'batch'
        # algo = ''
        # do this uing pool 

        # for i in range(50):
        #     run_behaviour(algo,aq,d,i,res_path)

        pool = mp.Pool(mp.cpu_count()-1)
        res = pool.starmap(run_behaviour, [(algo,aq,d,i,res_path) for i in range(50)])
        # results = [pool.apply(run_behaviour, args=(algo,aq,d,i,res_path)) for i in range(50)]
        pool.close()
        pool.join()
