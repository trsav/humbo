import sys
import os 
import jax.numpy as jnp
from jax import vmap
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.function import Function
import matplotlib.pyplot as plt 
from main import bo
from utils import * 
import uuid 

# ideas

# 1. lengthscale is constant (previous)
# 2. lengthscale is a prior (based on previous)
# 3. lengthscale distribution learned using EM 
# 4. heuristically favour smaller lengthscales



f = Function("function_creation/function.pkl")

def plot_function(f):
    x = jnp.linspace(f.bounds["x"][0],f.bounds["x"][1],500)
    y = jnp.array([f({"x":x_}) for x_ in x])
    fig,ax = plt.subplots(figsize=(8,2))
    ax.plot(x,y,c='k',lw=2)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([]); ax.set_xticks([])
    fig.tight_layout()
    plt.savefig("bo/function.png",dpi=400)
    return 


p_string = str(uuid.uuid4())
path = "bo/" + p_string + "/"

problem_data = {}
problem_data["sample_initial"] = 8
problem_data["ms_num"] = 16
problem_data["gp_ms"] = 8
problem_data["err_tol"] = 1e-4
problem_data['alternatives'] = 5
try:
    os.mkdir(path)
except:
    pass
bo(
    f,
    aq,
    path + "res.json",
    problem_data,
    path=path,
    printing=True,
)

