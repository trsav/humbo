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


problem_data_path = sys.argv[1]

problem_data = pd.read_csv(problem_data_path).to_dict(orient='records')[0]
try:
    df = pd.read_csv('bo/'+str(problem_data['behaviour'])+'_problems.csv')
except FileNotFoundError:
    df = pd.DataFrame({})

f_keys = pd.read_csv('function_creation/f_keys.csv')['f_keys'].values
problems = len(f_keys)
aqs = {'EI':EI,'UCB':UCB}

for i in range(problems):
    f_key = f_keys[i]
    key = random.PRNGKey(f_key)
    aq = problem_data['acquisition_function']
    f = Function(create_problem(key,problem_data['lengthscale']))
    file = str(uuid.uuid4())
    path = "bo/"+problem_data['behaviour']+'/' + file + "/"
    problem_data['time_created'] = str(datetime.datetime.now())
    problem_data['file_name'] = file
    problem_data['function_key'] = f_key

    if len(df) == 0:
        df = pd.DataFrame(problem_data,index=[0])
    else:
        df = pd.concat([df,pd.DataFrame(problem_data,index=[0])])
    
    df.to_csv('bo/'+str(problem_data['behaviour'])+'_problems.csv',index=False)

    os.mkdir(path)
    plot_function(f,path+"function.pdf")

    bo(
        f,
        aqs[aq],
        problem_data,
        path=path
    )
