from jax.config import config
import pandas as pd 
from jax import numpy as jnp
from jax import jit, value_and_grad, vmap
import matplotlib
import jax.random as random
from pymoo.core.problem import ElementwiseProblem, Problem
import os
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve
import numpy as np
import numpy.random as rnd
import jax
import gpjax as gpx
import optax as ox
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from jax.scipy.optimize import minimize
from tensorflow_probability.substrates import jax as tfp
import shutil 
from matplotlib import rc
from utils import * 
import matplotlib.font_manager
import matplotlib as mpl
import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from function_creation.function import *

from pathlib import Path

# fpath = Path(mpl.get_data_path(), "bo/plots/BerkeleyMono.ttf")
# plt.rcParams["font.family"] = "monospace"
# plt.rcParams["font.monospace"] = ["Berkeley Mono"]

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
rc('text', usetex=True)
fpath = 'Arial'

def plot_regret(problem_data,axs,c,directory,max_it,b_w,unc,noise): 

    try:
        func = problem_data['function']
        func_flag = True
    except:
        func_flag = False

    files = os.listdir(directory)
    problem_data_list = []
    for i in tqdm(range(len(files))):
        if func_flag != True:
            results = directory+'/'+files[i] + '/res.json'
            # open json
            with open(results, "r") as f:
                data = json.load(f)
            problem_data_list.append(data['problem_data'])
        else:
            if '.' not in files[i] and func in files[i]:
                results = directory+'/'+files[i] + '/res.json'
                # open json
                with open(results, "r") as f:
                    try:
                        data = json.load(f)
                        problem_data_list.append(data['problem_data'])
                    except:
                        print('ERROR IN JSON')

        
    # create dataframe from list of dictionaries 
    df = pd.DataFrame(problem_data_list)
    if noise > 0:
        print('here')
    if func_flag == False:
        df = df.loc[(df['human_behaviour'] == problem_data['human_behaviour']) & (df['dim'] == problem_data['dim']) & (np.abs(df['noise'] - noise) < 1e-8)]
    else:
        df = df.loc[(df['human_behaviour'] == problem_data['human_behaviour']) & (df['function'] == problem_data['function']) & (np.abs(df['noise'] - noise) < 1e-8)]


    label = problem_data['human_behaviour']
    file_names = df['file_name'].values
    regret_list, obj_list, f_opt_list = [], [], []
    for file in file_names:
        file = directory + '/' + file.split('/')[-2]
        data_full = read_json(file+'/res.json')
        data = data_full['data']
        f_opt = data_full['problem_data']['f_opt']
        obj = [d['objective'] for d in data]
        f_opt_list.append(f_opt)
        obj_list.append(obj)
        problem_data = data_full['problem_data']
    init = problem_data['sample_initial']
    full_it = problem_data['max_iterations']

    average_regret_list = []
    regret_list = []
    for obj,i in zip(obj_list,range(len(obj_list))):
        if len(obj) != full_it:
            obj += [obj[-1]]*(full_it-len(obj))

        it = len(obj)
        regret = [f_opt_list[i] - np.max(obj[:t]) for t in range(1,it+1)]
        for j in range(len(regret)):
            if regret[j] < 0: 
                regret[j] = 0 

        regret_list.append(regret)
        average_regret = [f_opt_list[i] - (1/t) * np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret_list.append(average_regret)

    max_it = min(len(regret_list[0]),max_it)
    regret_list = np.array(regret_list)[:,:max_it]
    average_regret = np.mean(np.array(average_regret_list),axis=0)[:max_it]
    average_regret_std = np.std(np.array(average_regret_list),axis=0)[:max_it]

    if label.__class__ != float:
        label = label[0].upper() + label[1:]
    else:
        label = '$p($Best$)=$'+str(label) 

    if label == 'Llmbo':
        label = 'LLMBO'
    try:
        x = np.arange(init,len(average_regret))
        if b_w != True:
            axs[1].plot(x,average_regret[init:],c=c,lw=1.5,label=label)
            if unc == True:
                axs[1].fill_between(x,average_regret[init:]-average_regret_std[init:],average_regret[init:]+average_regret_std[init:],alpha=0.1,color=c,lw=0)
        else:
            axs[1].plot(x,average_regret[init:],c='k',lw=1.5,ls=c,label=label)

        ax = axs[0]
        regret_list = np.array(regret_list)
        mean_instantaneous_regret = np.mean(regret_list,axis=0)
        std_instantaneous_regret = np.std(regret_list,axis=0)

        # for k in range(len(regret_list)):
        #     # clip std so that mean-std is never less than 10-4
        #     if mean_instantaneous_regret[k] - std_instantaneous_regret[k] < 10e-4:
        #         std_instantaneous_regret[k] = 0 


        x = np.arange(init,len(mean_instantaneous_regret))
        if b_w != True:
            ax.plot(x,mean_instantaneous_regret[init:],c=c,lw=1.5,label=label)
        else:
            ax.plot(x,mean_instantaneous_regret[init:],c='k',lw=1.5,ls=c,label=label)
        lower = mean_instantaneous_regret[init:]-std_instantaneous_regret[init:]
        lower[lower<10e-4] = 10e-4
        upper = mean_instantaneous_regret[init:]+std_instantaneous_regret[init:]
        if b_w != True and unc == True:
            ax.fill_between(x,lower,upper,alpha=0.1,color=c,lw=0)
    except:
        return problem_data['sample_initial']
    return problem_data['sample_initial']


def format_plot(fig,axs,s_i):
    fs = 12
    axs[0].set_ylabel(r"Simple Regret, $r_\tau$",fontsize=fs,font=fpath)
    for ax in axs:
        ax.grid(True,alpha=0.5)
        x_start = s_i
        max_y = ax.get_ylim()[1]
        min_y = ax.get_ylim()[0]
        ax.plot([x_start,x_start],[min_y,max_y],c='k',ls='--',lw=1,alpha=0.5)
    
    axs[0].set_xlabel(r"Iterations, $\tau$",fontsize=fs,font=fpath)
    axs[1].set_xlabel(r"Iterations, $\tau$",fontsize=fs,font=fpath)
    axs[1].set_ylabel(r"Average Regret, ${R_\tau}/{\tau}$",fontsize=fs,font=fpath)
    axs[0].set_yscale('log')
    # add text in upper right of right plot with functon name

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.825), ncol=6,frameon=False)


    #fig.suptitle(r'Regret expectation over 50 functions, $f \sim \mathcal{GP}(\mu \equiv 0, K_M (d,\nu = '+str(l)+'))$, '+str(problem_data['alternatives'])+' alternate choices, $\mathcal{U}(x)=$'+str(aq)+r', $x \in R^'+str(problem_data['dim'])+'$',fontsize=int(fs))
    fig.tight_layout()
    fig.subplots_adjust(top = 0.85)
    return fig,axs


def plot_rkhs(noise,d,max_it,b_w=False):
    directory = 'bo/benchmark_results_rkhs'
    human_behaviours = ['expert','adversarial','trusting',0.75,0.5,0.25]

    fig,axs = plt.subplots(1,2,figsize=(9,2.5))

    for i in range(len(human_behaviours)):
        # for this problem data
        problem_data = {}
        problem_data['dim'] = d
        # at a given human behaviour
        problem_data['human_behaviour'] = human_behaviours[i]
        problem_data['noise'] = noise
        
        if b_w == False:
            colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
            try:
                s_i = plot_regret(problem_data,axs,colors[i],directory,max_it,b_w,unc=True,noise=noise)
                plt.savefig('bo/plots/rkhs/d_'+str(d)+'_noise_'+str(noise)+'.pdf')
            except:
                pass
        if b_w == True:
            lines = ['-','--','-.',':',(0,1,10),(0, (3, 5, 1, 5, 1, 5))]
            try:
                s_i = plot_regret(problem_data,axs,lines[i],directory,max_it,b_w,unc=True,noise=noise)
                plt.savefig('bo/plots/rkhs/d_'+str(d)+'_noise_'+str(noise)+'.pdf')
            except:
                pass
    fig,axs = format_plot(fig,axs,s_i)
    try:
        plt.savefig('bo/plots/rkhs/d_'+str(d)+'_noise_'+str(noise)+'.pdf')
    except:
        os.mkdir('bo/plots')


def plot_specific(noise,max_it,b_w):
    directory = 'bo/benchmark_results_specific'
    colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
    human_behaviours = ['expert','trusting',0.75,0.5,0.25]
    
    functions = []

    for i in [2,3,5]:
        functions.append(Levi(i))
        functions.append(Schewefel(i))
        functions.append(Ackley(i))
        functions.append(Griewank(i))
        functions.append(Rastrigin(i))
        functions.append(Rosenbrock(i))

    # functions = [Ackley(3),Rosenbrock(2),Schewefel(3)]

    for function in functions:
        fig,axs = plt.subplots(1,2,figsize=(9,3))
        
        for i in range(len(human_behaviours)):
            # for this problem data
            problem_data = {}
            problem_data['human_behaviour'] = human_behaviours[i]
            problem_data['function'] = function.name
            
            noise_scaled = noise * jnp.abs(function.f_max - function.f_opt).item()

            if b_w == False:
                colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
                s_i = plot_regret(problem_data,axs,colors[i],directory,max_it,b_w,unc=False,noise=noise_scaled)
            if b_w == True:
                lines = ['-','--','-.',':',(0,1,10),(0, (3, 5, 1, 5, 1, 5))]
                try:
                    s_i = plot_regret(problem_data,axs,lines[i],directory,max_it,b_w,unc=False,noise=noise_scaled)
                except:
                    pass

        fs = 12
        if function.name.split('1')[-1] == '0':
            n = str(10)
            func_name = function.name.split('1')[0]
        else:
            n = function.name[-1]
            func_name = function.name[:-1]
        #axs[0].text(0.95, 0.95,func_name + ': $d= $'+n+', $\epsilon \sim \mathcal{N}(0,$'+str(noise)+r'$)$', horizontalalignment='right',verticalalignment='top', transform=axs[0].transAxes,fontsize=12,bbox=dict(facecolor='white',edgecolor='none',pad=0.5))
        # set title for both plots 
        fig.suptitle(func_name + ': $d= $'+n+', $\epsilon \sim \mathcal{N}(0,$'+str(noise)+r'$)$',x=0.5,y=0.1)

        fig,axs = format_plot(fig,axs,s_i)
        plt.savefig('bo/plots/specific/'+function.name+'_noise_'+str(noise)+'.png',dpi=400)


def plot_real(max_it,b_w):
    directory = 'bo/benchmark_results_real'
    colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
    human_behaviours = ['expert','trusting']

    functions = ['selfopt','reactor','AgNP','Crossed barrel','Perovskite']

    for function in functions:
        fig,axs = plt.subplots(1,2,figsize=(9,2.5))
        
        for i in range(len(human_behaviours)):
            # for this problem data
            problem_data = {}
            problem_data['human_behaviour'] = human_behaviours[i]
            problem_data['acquisition_function'] = 'UCB'
            problem_data['function'] = function
            if b_w == False:
                colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
                try:
                    s_i = plot_regret(problem_data,axs,colors[i],directory,max_it,b_w,unc=True)
                except:
                    pass
            if b_w == True:
                lines = ['-','--','-.',':',(0,1,10),(0, (3, 5, 1, 5, 1, 5))]
                try:
                    s_i = plot_regret(problem_data,axs,lines[i],directory,max_it,b_w,unc=True)
                except:
                    pass

        fs = 12

        func_str = function

        axs[1].text(0.95, 0.95,func_str , horizontalalignment='right',verticalalignment='top', transform=axs[1].transAxes,fontsize=12,bbox=dict(facecolor='white',edgecolor='none',pad=0.5))
        try:
            fig,axs = format_plot(fig,axs,s_i)
        except:
            pass
        plt.savefig('bo/plots/overall_regret_'+function+'.pdf')





# plot_human('EI',1)
b_w = False
for noise in [0.0,0.025,0.05]:
    # plot_rkhs(noise,1,1000,b_w)
    # plot_rkhs(noise,2,1000,b_w)
    # plot_rkhs(noise,3,1000,b_w)
    plot_specific(noise,10000,b_w)
#plot_real(50,b_w)
