import sys
import os
import jax.numpy as jnp
from jax import vmap
import datetime
import pandas as pd 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.function import SimpleFunction
import matplotlib.pyplot as plt
from algorithm import bo_human
from utils import *
import uuid
from function_creation.create_problem import create_problem
import multiprocessing as mp
import gc 
import resource
import numpy as np
from scipy.integrate import ode




def run_reaction_conditions(automated,iteration,path,name):
    # for this problem data
    aq = 'UCB'
    problem_data = {}
    problem_data["sample_initial"] = 8
    problem_data["gp_ms"] = 8
    problem_data["alternatives"] = 3
    problem_data["NSGA_iters"] = 50
    problem_data['max_iterations'] = 100
    problem_data['automated'] = automated
    problem_data['acquisition_function'] = aq

    aqs = {'EI':EI,'UCB':UCB}

    # for a given function...

    file = name
    path = path + file + "/"

    problem_data['time_created'] = str(datetime.datetime.now())
    problem_data['file_name'] = path

    def solve_ode(f, c0, t, k,x):
        k = np.append(k,x)
        c = np.zeros((len(t), len(c0)))
        c[0, :] = c0
        r = ode(f)
        r.set_initial_value(c[0], t[0]).set_f_params(k)

        for k in range(1, len(t)):
            c[k, :] = r.integrate(t[k])
            r.set_initial_value(c[k, :], t[k])
        return c


    def kinetic_model(t, s, k):
        s1,s2 = s
        k1,k2,E1,E2,R,a0,a1,a2,a3 = k
        T = a0 + a1*t + a2*t**2 + a3*t**3
        dsdt = [0 for i in range(len(s))]
        dsdt[0] = - k1*s1*np.exp(-E1/(R*T))
        dsdt[1] = k1*s1*np.exp(-E1/(R*T)) - k2*s2*np.exp(-E2/(R*T))
        return dsdt


    def f(x,ax):

        problem = {"f": kinetic_model, "name": "reaction"}
        problem["tf"] = 8
        problem["n"] = 600
        problem["x0"] = [0.7, 0]
        problem["param_names"] = ["k_1","k_2"]
        problem["var_names"] = ["$x_1$", "$x_2$"]
        problem["k"] = [1.335E10,1.149E17,75000,125000,8.31]

        f = problem["f"]
        tf = problem["tf"]
        n = problem["n"]
        x0 = problem["x0"]
        k = problem["k"]

        t = np.linspace(0, tf, n)
        y = solve_ode(f, x0, t, k, x)

        a0,a1,a2,a3 = x
        T = a0 + a1*t + a2*t**2 + a3*t**3

        # ax.set_ylabel('Concentration (mol/L)')
        # ax.set_xlabel('Time (min)')
        # ls = ['solid','--',':']
        # for i in range(len(y[0])):
        #     ax.plot(t, y[:, i], label=problem["var_names"][i],c='k',ls=ls[i])

        # ax.plot(t,T,c='tab:red',ls='solid',label='Temperature (K)')
        # plot on same axis, temperature on right yaxis, concentration on left yaxis
        ax.set_ylabel('Concentration (mol/L)')
        ax.set_xlabel('Time (min)')
        ls = ['solid','--',':']
        for i in range(len(y[0])):
            ax.plot(t, y[:, i], label=problem["var_names"][i],c='k',ls=ls[i])
        
        ax2 = ax.twinx()
        ax2.plot(t,T,c='tab:red',ls='solid',label='Temperature (K)')
        ax2.set_ylabel('Temperature (K)')
        # ax2.set_ylim([300,400])
        # ax2.set_xlim([0,8])
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.spines['right'].set_color('tab:red')
        ax2.yaxis.label.set_color('tab:red')
        ax2.tick_params(axis='y', colors='tab:red')
        ax2.set_zorder(1)
        ax.set_zorder(2)
        ax.patch.set_visible(False)
        ax2.patch.set_visible(False)


        # left yaxis = concenration 
        ax.grid(alpha=0.5)

        # lgend but outside of plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # ax.legend(frameon=False)

        return y[-1,1],ax

    def visualise(ax,x):
        
        t = np.linspace(0, 8, 600)
        temperature_path = x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3
        ax.plot(t,temperature_path,c='k')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Temperature (K)')

        return ax

    f = SimpleFunction(f,[[325,375],[-3,3],[-0.2,0.2],[-0.1,0.1]])


    bo_human(
        f,
        aqs[aq],
        problem_data,
        visualise
    )

try:
    index = sys.argv[1]
except:
    index = 0 
path = 'bo/human_reaction_conditions_results/'
try:
    os.mkdir(path)
except:
    pass
name = 'tom'
run_reaction_conditions(False,index,path,name)