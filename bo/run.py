import sys
import os
import jax.numpy as jnp
from jax import vmap
import datetime
import pandas as pd 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.function import Function
import matplotlib.pyplot as plt
from main import bo
from utils import *
import uuid
from function_creation.create_problem import create_problem

def num_split(n,f):
    return n%f,n//f

f_keys = pd.read_csv('function_creation/f_keys.csv')['f_keys'].values
problems = len(f_keys)
human_behaviours = ['expert','adversarial','trusting',0.25,0.5,0.75]
try:
    index = int(sys.argv[1])
    f_index, behaviour_index = num_split(index,problems)
except:
    f_index = 0
    behaviour_index = 0

# for this problem data
aq = 'UCB'
problem_data = {}
problem_data["sample_initial"] = 4
problem_data["gp_ms"] = 8
problem_data["alternatives"] = 3
problem_data["NSGA_iters"] = 50
problem_data["plotting"] = False
problem_data['regret_tolerance'] = 0.0001
problem_data['max_iterations'] = 75
problem_data['lengthscale'] = 0.4
# at a given human behaviour
problem_data['human_behaviour'] = human_behaviours[behaviour_index]
problem_data['acquisition_function'] = aq


aqs = {'EI':EI,'UCB':UCB}

# for a given function...
f_key = f_keys[f_index]
key = random.PRNGKey(f_key)
aq = problem_data['acquisition_function']
f = Function(create_problem(key,problem_data['lengthscale']))
file = str(uuid.uuid4())
path = "bo/" + file + "/"
os.mkdir(path)

problem_data['time_created'] = str(datetime.datetime.now())
problem_data['file_name'] = path
problem_data['function_key'] = str(f_key)

# plot_function(f,path+"function.pdf")

bo(
    f,
    aqs[aq],
    problem_data,
    path=path
)
