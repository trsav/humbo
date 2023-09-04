import sys
import os
import jax.numpy as jnp
from jax import vmap
import datetime
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.function import Function
import matplotlib.pyplot as plt
from main import bo
from utils import *
import uuid
from function_creation.create_problem import create_problem

def plot_function(f,path):
    x = jnp.linspace(f.bounds["x"][0], f.bounds["x"][1], 500)
    y = f.eval_vector(x)
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(x, y, c="k", lw=2)
    x_opt = x[jnp.argmax(y)]
    y_opt = jnp.max(y)

    ax.scatter(x_opt,y_opt, c="k",marker='+', s=50,label='Global Optimum')
    # plotting a line from the optimum to the x-axis
    ax.plot([x_opt,x_opt],[y_opt,jnp.min(y)], c="k", lw=2,linestyle='--',alpha=0.5)
    ax.plot([f.bounds["x"][0],f.bounds["x"][1]],[y_opt,y_opt], c="k", lw=2,linestyle='--',alpha=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.savefig(path, dpi=400)
    return





for i in range(20):
    f_key = np.random.randint(0, 10000)
    key = random.PRNGKey(f_key)
    f = Function(create_problem(key))
    problem_data = {}
    problem_data['time_created'] = str(datetime.datetime.now())
    problem_data["sample_initial"] = 4
    problem_data["gp_ms"] = 8
    problem_data["alternatives"] = 3
    problem_data["NSGA_iters"] = 100
    problem_data["plotting"] = False
    problem_data['regret_tolerance'] = 0.05
    problem_data['max_iterations'] = 50
    problem_data['function_key'] = f_key
    problem_data['human_behaviour'] = 'trusting'

    file = str(uuid.uuid4())
    path = "bo/" + file + "/"
    os.mkdir(path)
    plot_function(f,path+"function.png")


    bo(
        f,
        EI,
        problem_data,
        path=path
    )
