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
from function_creation.ce_functions import * 

from pathlib import Path

# fpath = Path(mpl.get_data_path(), "bo/plots/BerkeleyMono.ttf")
# plt.rcParams["font.family"] = "monospace"
# plt.rcParams["font.monospace"] = ["Berkeley Mono"]

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
rc('text', usetex=True)
fpath = 'Arial'

def plot_regret(problem_data,axs,c,directory,max_it,b_w,unc,noise): 

    hum_mean = problem_data['hum_mean']
    try:
        func = problem_data['function']
        func_flag = True
    except:
        func_flag = False
        func = None

    files = os.listdir(directory)
    problem_data_list = []
    for i in tqdm(range(len(files))):

        if func == 'bioprocess_profile' or func == 'reactor':

            results = directory+'/'+files[i] + '/res.json'
            # open json
            with open(results, "r") as f:
                try:
                    data = json.load(f)
                    problem_data_list.append(data['problem_data'])
                except:
                    print('ERROR IN JSON')

        elif func_flag != True:
            results = directory+'/'+files[i] + '/res.json'
            # open json
            try:
                with open(results, "r") as f:
                    data = json.load(f)
                problem_data_list.append(data['problem_data'])
            except:
                print('ERROR IN JSON')
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

    if noise == None:
        df = df.loc[(df['human_behaviour'] == problem_data['human_behaviour'])]
    elif func_flag == False:
        df = df.loc[(df['human_behaviour'] == problem_data['human_behaviour']) & (df['dim'] == problem_data['dim']) & (np.abs(df['noise'] - noise) < 1e-8)]
    else:
        df = df.loc[(df['human_behaviour'] == problem_data['human_behaviour']) & (df['function'] == problem_data['function']) & (np.abs(df['noise'] - noise) < 1e-8)]

    aq_bool_mask = [problem_data['aq'] in df_item for df_item in df['acquisition_function']]
    df = df.loc[aq_bool_mask]


    print(df)
    label = problem_data['human_behaviour']
    file_names = df['file_name'].values
    regret_list, obj_list, f_opt_list, name_list = [], [], [], []
    for file in file_names:
        if func != 'bioprocess_profile' and func != 'reactor':
            file = directory + '/' + file.split('/')[-2]

        data_full = read_json(file+'/res.json')
        data = data_full['data']
        f_opt = data_full['problem_data']['f_opt']
        obj = [d['objective'] for d in data]
        f_opt_list.append(f_opt)
        obj_list.append(obj)
        name_list.append(data_full['problem_data']['file_name'].split('/')[-2])
        problem_data = data_full['problem_data']
    init = problem_data['sample_initial']
    if func == 'bioprocess_profile' or func == 'reactor':
        init = 0 

    full_it = problem_data['max_iterations']

    average_regret_list = []
    regret_list = []
    for obj,i in zip(obj_list,range(len(obj_list))):

        if len(obj) != full_it:
            obj += [obj[-1]]*(full_it-len(obj))

        it = len(obj)
        
        if func != 'bioprocess_profile' and func != 'reactor':
            regret = [f_opt_list[i] - np.max(obj[:t]) for t in range(1,it+1)]
        else:
            regret = [np.max(obj[:t]) for t in range(1,it+1)]
            # for j in range(len(regret)):
            #     if regret[j] < 0: 
            #         regret[j] = 0 

        # if func == 'reactor':
        #     for j in range(len(regret)):
        #         regret[j] -= 20
        # if label == 'trusting' and noise < 0.001 and problem_data['dim'] == 1:
        #     if regret[-1] < 0.1:
        #         regret_list.append(regret)
        #         average_regret = [f_opt_list[i] - (1/t) * np.sum(obj[:t]) for t in range(1,it+1)]
        #         average_regret_list.append(average_regret)
        # else:
        
        regret_list.append(regret)
        if func != 'bioprocess_profile' and func != 'reactor':
            average_regret = [f_opt_list[i] - (1/t) * np.sum(obj[:t]) for t in range(1,it+1)]
        else:
            average_regret = [(1/t) * np.sum(obj[:t]) for t in range(1,it+1)]

        # if func == 'reactor':
        #     for j in range(len(average_regret)):
        #         average_regret[j] -= 20
        average_regret_list.append(average_regret)



    # fig_reg,ax_reg = plt.subplots(1,1)
    # ax_reg.imshow(np.array(regret_list))
    # ax_reg.colorbar()
    # fig_reg.savefig('bo/plots/regret_heatmap_'+str(label)+'.pdf')
    # fig_reg.close()

    max_it = min(len(regret_list[0]),max_it)
    regret_list = np.array(regret_list)[:,:max_it]
    average_regret = np.mean(np.array(average_regret_list),axis=0)[:max_it]
    average_regret_std = np.std(np.array(average_regret_list),axis=0)[:max_it]

    

    # if func == 'bioprocess_profile' or func == 'reactor':

    #     # make regret positive (turn into a reward)

    #     new_regret_list = []
    #     for regret in regret_list:
    #         new_regret_list.append(list(-np.array(regret)))
    #         regret_list = new_regret_list

    #     new_average_regret_list = []
    #     for average_regret_ in average_regret_list:
    #         new_average_regret_list.append(list(-np.array(average_regret_)))
        
    #     average_regret_list = new_average_regret_list
    #     # same with average regret
    #     average_regret = -average_regret

    if label.__class__ != float:
        label = label[0].upper() + label[1:]
    else:
        label = '$p($Best$)=$'+str(label) 

    if label == 'Llmbo':
        label = 'LLMBO'
    try:
        x = np.arange(init,len(average_regret))
        if problem_data['human_behaviour'] != 'human' or hum_mean == True:
            axs[1].plot(x,average_regret[init:],c=c,lw=1.5,label=label)
            if unc == True:
                axs[1].fill_between(x,average_regret[init:]-average_regret_std[init:],average_regret[init:]+average_regret_std[init:],alpha=0.1,color=c,lw=0)
        else:
            # for average_regret,i in enumerate(average_regret_list,range(len(average_regret_list))):
            colors = ['tab:blue','tab:green','tab:orange','tab:purple','tab:brown'] 
            for i in range(len(average_regret_list)):
                average_regret = average_regret_list[i]
                name = name_list[i][0].upper() + name_list[i][1:]
                axs[1].plot(x,average_regret[init:],c=colors[i],ls='--',lw=1.5,alpha=0.75,label=name)


        ax = axs[0]
        regret_list = np.array(regret_list)
        mean_instantaneous_regret = np.mean(regret_list,axis=0)
        std_instantaneous_regret = np.std(regret_list,axis=0)

        # for k in range(len(regret_list)):
        #     # clip std so that mean-std is never less than 10-4
        #     if mean_instantaneous_regret[k] - std_instantaneous_regret[k] < 10e-4:
        #         std_instantaneous_regret[k] = 0 


        x = np.arange(init,len(mean_instantaneous_regret))

        if problem_data['human_behaviour'] != 'human' or hum_mean == True:
            ax.plot(x,mean_instantaneous_regret[init:],c=c,lw=1.5,label=label)
            lower = mean_instantaneous_regret[init:]-std_instantaneous_regret[init:]
            if func != 'bioprocess_profile' or func != 'reactor':
                lower[lower<10e-4] = 10e-4
            upper = mean_instantaneous_regret[init:]+std_instantaneous_regret[init:]
            if b_w != True and unc == True:
                ax.fill_between(x,lower,upper,alpha=0.1,color=c,lw=0)
        else:
            colors = ['tab:blue','tab:green','tab:orange','tab:purple','tab:brown'] 
            for i in range(len(regret_list)):
                regret = regret_list[i] 
                name = name_list[i][0].upper() + name_list[i][1:]
                ax.plot(x,regret[init:],c=colors[i],ls='--',alpha=0.75,lw=1.5,label=name)

    except:
        return problem_data['sample_initial']
    return problem_data['sample_initial']


def plot_simple_regret(problem_data,ax,c,directory,max_it,label,b_w,unc,noise): 

    hum_mean = problem_data['hum_mean']
    try:
        func = problem_data['function']
        func_flag = True
    except:
        func_flag = False
        func = None

    files = os.listdir(directory)
    problem_data_list = []
    for i in tqdm(range(len(files))):

        if func == 'bioprocess_profile' or func == 'reactor':

            results = directory+'/'+files[i] + '/res.json'
            # open json
            with open(results, "r") as f:
                try:
                    data = json.load(f)
                    problem_data_list.append(data['problem_data'])
                except:
                    print('ERROR IN JSON')

        elif func_flag != True:
            results = directory+'/'+files[i] + '/res.json'
            # open json
            try:
                with open(results, "r") as f:
                    data = json.load(f)
                problem_data_list.append(data['problem_data'])
            except:
                print('ERROR IN JSON')
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

    if noise == None:
        df = df.loc[(df['human_behaviour'] == problem_data['human_behaviour'])]
    elif func_flag == False:
        df = df.loc[(df['human_behaviour'] == problem_data['human_behaviour']) & (df['dim'] == problem_data['dim']) & (np.abs(df['noise'] - noise) < 1e-8)]
    else:
        df = df.loc[(df['human_behaviour'] == problem_data['human_behaviour']) & (df['function'] == problem_data['function']) & (np.abs(df['noise'] - noise) < 1e-8)]

    aq_bool_mask = [problem_data['aq'] in df_item for df_item in df['acquisition_function']]
    df = df.loc[aq_bool_mask]


    print(df)
    label = problem_data['human_behaviour']
    file_names = df['file_name'].values
    regret_list, obj_list, f_opt_list, name_list = [], [], [], []
    for file in file_names:
        if func != 'bioprocess_profile' and func != 'reactor':
            file = directory + '/' + file.split('/')[-2]

        data_full = read_json(file+'/res.json')
        data = data_full['data']
        f_opt = data_full['problem_data']['f_opt']
        obj = [d['objective'] for d in data]
        f_opt_list.append(f_opt)
        obj_list.append(obj)
        name_list.append(data_full['problem_data']['file_name'].split('/')[-2])
        problem_data = data_full['problem_data']
    init = problem_data['sample_initial']
    if func == 'bioprocess_profile' or func == 'reactor':
        init = 0 

    full_it = problem_data['max_iterations']

    average_regret_list = []
    regret_list = []
    for obj,i in zip(obj_list,range(len(obj_list))):

        if len(obj) != full_it:
            obj += [obj[-1]]*(full_it-len(obj))

        it = len(obj)
        
        if func != 'bioprocess_profile' and func != 'reactor':
            regret = [f_opt_list[i] - np.max(obj[:t]) for t in range(1,it+1)]
        else:
            regret = [np.max(obj[:t]) for t in range(1,it+1)]

        regret_list.append(regret)

    max_it = min(len(regret_list[0]),max_it)
    regret_list = np.array(regret_list)[:,:max_it]

    
    if label.__class__ != float:
        label = label[0].upper() + label[1:]
    else:
        label = '$p($Best$)=$'+str(label) 

    if label == 'Llmbo':
        label = 'LLMBO'

    x = np.arange(init,len(regret_list[0]))

    regret_list = np.array(regret_list)
    mean_instantaneous_regret = np.mean(regret_list,axis=0)
    std_instantaneous_regret = np.std(regret_list,axis=0)


    x = np.arange(init,len(mean_instantaneous_regret))

    if label:
        ax.plot(x,mean_instantaneous_regret[init:],c=c,lw=1.5,label=label)
    else: 
        ax.plot(x,mean_instantaneous_regret[init:],c=c,lw=1.5)
    lower = mean_instantaneous_regret[init:]-std_instantaneous_regret[init:]
    upper = mean_instantaneous_regret[init:]+std_instantaneous_regret[init:]
    # if unc == True:
    #     ax.fill_between(x,lower,upper,alpha=0.1,color=c,lw=0)
    # else:
    #     colors = ['tab:blue','tab:green','tab:orange','tab:purple','tab:brown'] 
    #     for i in range(len(regret_list)):
    #         regret = regret_list[i] 
    #         name = name_list[i][0].upper() + name_list[i][1:]
    #         ax.plot(x,regret[init:],c=colors[i],ls='--',alpha=0.75,lw=1.5,label=name)

    return ax


def plot_simple_regret_alts(problem_data,ax,c,directory,max_it,label,b_w,unc,noise): 

    hum_mean = problem_data['hum_mean']
    try:
        func = problem_data['function']
        func_flag = True
    except:
        func_flag = False
        func = None

    files = os.listdir(directory)
    problem_data_list = []
    for i in tqdm(range(len(files))):

        if func == 'bioprocess_profile' or func == 'reactor':

            results = directory+'/'+files[i] + '/res.json'
            # open json
            with open(results, "r") as f:
                try:
                    data = json.load(f)
                    problem_data_list.append(data['problem_data'])
                except:
                    print('ERROR IN JSON')

        elif func_flag != True:
            results = directory+'/'+files[i] + '/res.json'
            # open json
            try:
                with open(results, "r") as f:
                    data = json.load(f)
                problem_data_list.append(data['problem_data'])
            except:
                print('ERROR IN JSON')
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

    df = df.loc[(df['human_behaviour'] == problem_data['human_behaviour']) &  (df['alternatives'] == problem_data['alternatives'])]

    aq_bool_mask = [problem_data['aq'] in df_item for df_item in df['acquisition_function']]
    df = df.loc[aq_bool_mask]


    print(df)
    label = problem_data['human_behaviour']
    file_names = df['file_name'].values
    regret_list, obj_list, f_opt_list, name_list = [], [], [], []
    for file in file_names:
        if func != 'bioprocess_profile' and func != 'reactor':
            file = directory + '/' + file.split('/')[-2]

        data_full = read_json(file+'/res.json')
        data = data_full['data']
        f_opt = data_full['problem_data']['f_opt']
        obj = [d['objective'] for d in data]
        f_opt_list.append(f_opt)
        obj_list.append(obj)
        name_list.append(data_full['problem_data']['file_name'].split('/')[-2])
        problem_data = data_full['problem_data']
    init = problem_data['sample_initial']
    if func == 'bioprocess_profile' or func == 'reactor':
        init = 0 

    full_it = problem_data['max_iterations']

    average_regret_list = []
    regret_list = []
    for obj,i in zip(obj_list,range(len(obj_list))):

        if len(obj) != full_it:
            obj += [obj[-1]]*(full_it-len(obj))

        it = len(obj)
        
        if func != 'bioprocess_profile' and func != 'reactor':
            regret = [f_opt_list[i] - np.max(obj[:t]) for t in range(1,it+1)]
        else:
            regret = [np.max(obj[:t]) for t in range(1,it+1)]

        regret_list.append(regret)

    max_it = min(len(regret_list[0]),max_it)
    regret_list = np.array(regret_list)[:,:max_it]

    
    if label.__class__ != float:
        label = label[0].upper() + label[1:]
    else:
        label = '$p($Best$)=$'+str(label) 

    if label == 'Llmbo':
        label = 'LLMBO'

    x = np.arange(init,len(regret_list[0]))

    regret_list = np.array(regret_list)
    mean_instantaneous_regret = np.mean(regret_list,axis=0)
    std_instantaneous_regret = np.std(regret_list,axis=0)


    x = np.arange(init,len(mean_instantaneous_regret))

    if label:
        ax.plot(x,mean_instantaneous_regret[init:],c=c,lw=1.5,label=label)
    else: 
        ax.plot(x,mean_instantaneous_regret[init:],c=c,lw=1.5)
    lower = mean_instantaneous_regret[init:]-std_instantaneous_regret[init:]
    upper = mean_instantaneous_regret[init:]+std_instantaneous_regret[init:]
    # if unc == True:
    #     ax.fill_between(x,lower,upper,alpha=0.1,color=c,lw=0)
    # else:
    #     colors = ['tab:blue','tab:green','tab:orange','tab:purple','tab:brown'] 
    #     for i in range(len(regret_list)):
    #         regret = regret_list[i] 
    #         name = name_list[i][0].upper() + name_list[i][1:]
    #         ax.plot(x,regret[init:],c=colors[i],ls='--',alpha=0.75,lw=1.5,label=name)

    return ax


def format_plot(fig,axs,s_i,type='Benchmark'):
    fs = 12
    for ax in axs:
        ax.grid(True,alpha=0.5)
        x_start = s_i
        max_y = ax.get_ylim()[1]
        min_y = ax.get_ylim()[0]
        ax.plot([x_start,x_start],[min_y,max_y],c='k',ls='--',lw=1,alpha=0.5)
    
    axs[0].set_xlabel(r"Iterations, $\tau$",fontsize=fs,font=fpath)
    axs[1].set_xlabel(r"Iterations, $\tau$",fontsize=fs,font=fpath)
    if type == 'Benchmark':
        axs[0].set_ylabel(r"Simple Regret, $r_\tau$",fontsize=fs,font=fpath)
        axs[1].set_ylabel(r"Average Regret, ${R_\tau}/{\tau}$",fontsize=fs,font=fpath)
        # axs[0].set_yscale('log')
    else:
        axs[0].set_ylabel(r"Simple Reward, $r_\tau$",fontsize=fs,font=fpath)
        axs[1].set_ylabel(r"Average Reward, ${R_\tau}/{\tau}$",fontsize=fs,font=fpath)

    # add text in upper right of right plot with functon name

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.825), ncol=6,frameon=False)


    #fig.suptitle(r'Regret expectation over 50 functions, $f \sim \mathcal{GP}(\mu \equiv 0, K_M (d,\nu = '+str(l)+'))$, '+str(problem_data['alternatives'])+' alternate choices, $\mathcal{U}(x)=$'+str(aq)+r', $x \in R^'+str(problem_data['dim'])+'$',fontsize=int(fs))
    fig.tight_layout()
    fig.subplots_adjust(top = 0.85)
    return fig,axs


def plot_rkhs(aq,noise,d,max_it,b_w=False):
    directory = 'bo/benchmark_results_rkhs'
    human_behaviours = ['expert','adversarial','trusting',0.75,0.5,0.25]
    

    fig,axs = plt.subplots(1,2,figsize=(9,2.5))

    for i in range(len(human_behaviours)):
        # for this problem data
        problem_data = {}
        problem_data['dim'] = d
        problem_data['aq'] = aq
        # at a given human behaviour
        problem_data['human_behaviour'] = human_behaviours[i]
        problem_data['hum_mean'] = True
        problem_data['noise'] = noise
        
        colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
        try:
            s_i = plot_regret(problem_data,axs,colors[i],directory,max_it,b_w,unc=True,noise=noise)
        except:
            pass


    fig,axs = format_plot(fig,axs,s_i)
    try:
        # plt.savefig('bo/plots/rkhs/d_'+str(d)+'_noise_'+str(noise)+'_'+aq+'.pdf')
        plt.savefig('bo/plots/rkhs/d_'+str(d)+'_noise_'+str(noise)+'_'+aq+'UNC.pdf')
    except:
        os.mkdir('bo/plots')


def plot_specific(aq,noise,max_it,b_w):
    directory = 'bo/benchmark_results_specific'
    colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
    human_behaviours = ['expert','adversarial','trusting',0.75,0.5,0.25]
    
    functions = []

    # for i in [2,3,5]:
    #     functions.append(Levi(i))
    #     functions.append(Schwefel(i))
    #     functions.append(Ackley(i))
    #     functions.append(Griewank(i))
    #     functions.append(Rastrigin(i))
    #     functions.append(Rosenbrock(i))

    for i in [2,5,10]:
        # functions.append(Levi(i))
        functions.append(Schwefel(i))
        functions.append(Ackley(i))
        # functions.append(Griewank(i))
        # functions.append(Rastrigin(i))
        functions.append(Rosenbrock(i))

    # functions = [Ackley(3),Rosenbrock(2),Schewefel(3)]

    for function in functions:
        fig,axs = plt.subplots(1,2,figsize=(9,3))
        
        for i in range(len(human_behaviours)):
            # for this problem data
            problem_data = {}
            problem_data['human_behaviour'] = human_behaviours[i]
            problem_data['aq'] = aq
            problem_data['hum_mean'] = True
            problem_data['function'] = function.name
            
            noise_scaled = noise * jnp.abs(function.f_max - function.f_opt).item()

            colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
            try:
                s_i = plot_regret(problem_data,axs,colors[i],directory,max_it,b_w,unc=False,noise=noise_scaled)
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
        plt.savefig('bo/plots/specific/'+function.name+'_noise_'+str(noise)+'_'+str(aq)+'.pdf',dpi=400)


def plot_specific_grid(aq,noise,max_it,b_w):
    directory = 'bo/benchmark_results_specific'
    colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
    human_behaviours = ['expert','trusting',0.75,0.5,0.25]
    
    functions = []


    for i in [2,5,10]:
        functions.append(Schwefel(i))
    for i in [2,5,10]:
        functions.append(Ackley(i))
    for i in [2,5,10]:
        functions.append(Rosenbrock(i))

    fig,axs = plt.subplots(3,3,figsize=(7,6))

    k = 0 


    for i in range(3):
        axs[i,0].set_ylabel(r'Simple Regret, $r_\tau$')
        axs[2,i].set_xlabel(r'Iterations, $\tau$')

    k = 0 
    plt.subplots_adjust(wspace=0.3,hspace=0.3,top=0.875,bottom=0.1,left=0.1,right=0.975)
    for ax in axs.ravel():

        function = functions[k]
        k += 1

        # remove number from function name (only alphabet)
        function_name = ''.join([i for i in function.name if not i.isdigit()])
        function_name += ': ' + str(function.dim) + 'D'
        ax.set_title(function_name,fontsize=8)
        
        for i in range(len(human_behaviours)):
            # for this problem data
            problem_data = {}
            problem_data['human_behaviour'] = human_behaviours[i]
            problem_data['aq'] = aq
            problem_data['hum_mean'] = True
            problem_data['function'] = function.name
            
            noise_scaled = noise * jnp.abs(function.f_max - function.f_opt).item()

            colors = ['tab:red','tab:green','tab:orange','tab:purple','tab:brown']
            if k == 2:
                label = True
            else:
                label = False
            try:
                ax = plot_simple_regret(problem_data,ax,colors[i],directory,max_it,b_w,label,unc=False,noise=noise_scaled)

            except:
                pass
        # set the title within the plot itself 
        
        ax.grid(alpha=0.5)
        y_vals = np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],3)
        y_val_formatted = [f'{y:.1f}' for y in y_vals]
        ax.set_yticks(y_vals,y_val_formatted,fontsize=7)
        if k == 2:
            lines, labels = ax.get_legend_handles_labels()
            fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=6,frameon=False)

        plt.savefig('bo/plots/specific/grid.png',dpi=100)

        # fs = 12
        # if function.name.split('1')[-1] == '0':
        #     n = str(10)
        #     func_name = function.name.split('1')[0]
        # else:
        #     n = function.name[-1]
        #     func_name = function.name[:-1]
        #axs[0].text(0.95, 0.95,func_name + ': $d= $'+n+', $\epsilon \sim \mathcal{N}(0,$'+str(noise)+r'$)$', horizontalalignment='right',verticalalignment='top', transform=axs[0].transAxes,fontsize=12,bbox=dict(facecolor='white',edgecolor='none',pad=0.5))
        # set title for both plots 
        # fig.suptitle(func_name + ': $d= $'+n+', $\epsilon \sim \mathcal{N}(0,$'+str(noise)+r'$)$',x=0.5,y=0.1)

        # fig,axs = format_plot(fig,axs,s_i)
    
    plt.savefig('bo/plots/specific/grid.png',dpi=400)
    plt.savefig('bo/plots/specific/grid.pdf',dpi=400)


def plot_noise_grid(aq,max_it,b_w):

    directory = 'bo/benchmark_results_rkhs'
    human_behaviours = ['expert','adversarial','trusting',0.75,0.5,0.25]
    
    colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
    
    fig,axs = plt.subplots(2,3,figsize=(8,5))
    k = 0 

    for i in range(3):
        axs[1,i].set_xlabel(r'Iterations, $\tau$')
    for i in range(2):
        axs[i,0].set_ylabel(r'Simple Regret, $r_\tau$')

    dims = [2,3]

    noises = [0,0.05,0.1]
    plt.subplots_adjust(wspace=0.15,hspace=0.3,top=0.875,bottom=0.1,left=0.075,right=0.975)
    for k in range(2):
        for j in range(3):
            ax = axs[k,j]
            dim = dims[k]
            noise = noises[j]

            ax.set_title(str(dim)+'D'+', Noise: '+str(int(noise*100))+r"$\%$",fontsize=8)

            # # remove number from function name (only alphabet)
            # function_name = ''.join([i for i in function.name if not i.isdigit()])
            # function_name += ': ' + str(function.dim) + 'D'
            # ax.set_title(function_name,fontsize=8)
            
            for i in range(len(human_behaviours)):
                # for this problem data
                problem_data = {}
                problem_data['human_behaviour'] = human_behaviours[i]
                problem_data['dim'] = dim
                problem_data['aq'] = aq
                problem_data['hum_mean'] = True
                # problem_data['function'] = function.name
                
                # noise_scaled = noise * jnp.abs(function.f_max - function.f_opt).item()

                # colors = ['tab:red','tab:green','tab:orange','tab:purple','tab:brown']
                if k == 0 and j == 1:
                    label = True
                else:
                    label = False
                try:
                    ax = plot_simple_regret(problem_data,ax,colors[i],directory,max_it,b_w,label,unc=False,noise=noise)

                except:
                    pass
            # set the title within the plot itself 
            
            ax.grid(alpha=0.5)
            # ax.set_yticklabels([])
            # set three equally spaced y axis ticks based on thte max and min of the data
            y_vals = np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],3)
            y_val_formatted = [f'{y:.2f}' for y in y_vals]
            ax.set_yticks(y_vals,y_val_formatted,fontsize=7)
            if k ==0 and j == 1:
                lines, labels = ax.get_legend_handles_labels()
                fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=6,frameon=False)

            plt.savefig('bo/plots/rkhs/grid.png',dpi=100)

        # fs = 12
        # if function.name.split('1')[-1] == '0':
        #     n = str(10)
        #     func_name = function.name.split('1')[0]
        # else:
        #     n = function.name[-1]
        #     func_name = function.name[:-1]
        #axs[0].text(0.95, 0.95,func_name + ': $d= $'+n+', $\epsilon \sim \mathcal{N}(0,$'+str(noise)+r'$)$', horizontalalignment='right',verticalalignment='top', transform=axs[0].transAxes,fontsize=12,bbox=dict(facecolor='white',edgecolor='none',pad=0.5))
        # set title for both plots 
        # fig.suptitle(func_name + ': $d= $'+n+', $\epsilon \sim \mathcal{N}(0,$'+str(noise)+r'$)$',x=0.5,y=0.1)

        # fig,axs = format_plot(fig,axs,s_i)
    
    plt.subplots_adjust(wspace=0.15,hspace=0.3,top=0.875,bottom=0.1,left=0.075,right=0.975)
    plt.savefig('bo/plots/rkhs/grid.png',dpi=400)
    plt.savefig('bo/plots/rkhs/grid.pdf',dpi=400)

def plot_alt_grid(max_it,b_w):

    directory = 'bo/benchmark_results_specific_alternatives'
    human_behaviours = ['expert','adversarial','trusting',0.75,0.5,0.25]
    
    colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
    
    alts = [3,4,5,6]
    fig,axs = plt.subplots(1,len(alts) ,figsize=(8,2),sharey=True)
    k = 0 

    for i in range(len(alts)):
        axs[i].set_xlabel(r'Iterations, $\tau$')
    axs[0].set_ylabel(r'Simple Regret, $r_\tau$')


    noises = [0.025]
    for j in range(len(alts)):
        alt = alts[j]
        ax = axs[j]
        f = Ackley(2)
        dim = 2
        noise = 0.025

        ax.set_title(f"{alt} Alternatives",fontsize=8)

        # # remove number from function name (only alphabet)
        # function_name = ''.join([i for i in function.name if not i.isdigit()])
        # function_name += ': ' + str(function.dim) + 'D'
        # ax.set_title(function_name,fontsize=8)
        
        for i in range(len(human_behaviours)):
            # for this problem data
            problem_data = {}
            problem_data['human_behaviour'] = human_behaviours[i]
            problem_data['function'] = f.name
            problem_data['aq'] = aq
            problem_data['hum_mean'] = True
            problem_data['alternatives'] = alt
            # problem_data['function'] = function.name
            
            # noise_scaled = noise * jnp.abs(function.f_max - function.f_opt).item()

            # colors = ['tab:red','tab:green','tab:orange','tab:purple','tab:brown']
            if k == 0 and j == 1:
                label = True
            else:
                label = False
            try:
                ax = plot_simple_regret_alts(problem_data,ax,colors[i],directory,max_it,b_w,label,unc=False,noise=noise)

            except:
                pass
        # set the title within the plot itself 
        
        ax.grid(alpha=0.5)
        ax.set_yticklabels([])
        if j == 1:
            lines, labels = ax.get_legend_handles_labels()
            fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.85), ncol=6,frameon=False)
        y_vals = np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],3)
        y_val_formatted = [f'{y:.2f}' for y in y_vals]
        ax.set_yticks(y_vals,y_val_formatted,fontsize=7)

        plt.savefig('bo/plots/specific/grid_alts.png',dpi=100)

        # fs = 12
        # if function.name.split('1')[-1] == '0':
        #     n = str(10)
        #     func_name = function.name.split('1')[0]
        # else:
        #     n = function.name[-1]
        #     func_name = function.name[:-1]
        #axs[0].text(0.95, 0.95,func_name + ': $d= $'+n+', $\epsilon \sim \mathcal{N}(0,$'+str(noise)+r'$)$', horizontalalignment='right',verticalalignment='top', transform=axs[0].transAxes,fontsize=12,bbox=dict(facecolor='white',edgecolor='none',pad=0.5))
        # set title for both plots 
        # fig.suptitle(func_name + ': $d= $'+n+', $\epsilon \sim \mathcal{N}(0,$'+str(noise)+r'$)$',x=0.5,y=0.1)

        # fig,axs = format_plot(fig,axs,s_i)
    
    plt.subplots_adjust(wspace=0.075,hspace=0.3,top=0.825,bottom=0.2,left=0.075,right=0.975)
    plt.savefig('bo/plots/specific/grid_alts.png',dpi=400)
    plt.savefig('bo/plots/specific/grid_alts.pdf',dpi=400)


def plot_bioproc():
    directories = ['bo/bioprocess_profile','bo/bioprocess_profile_human']
    colors = ['tab:red','k']
    human_behaviours = ['trusting','human']
    
    functions = [BioProcess_Profile(4)]

    
    for hum in [True,False]:
        for function in functions:
            fig,axs = plt.subplots(1,2,figsize=(9,3))
            
            for i in range(len(human_behaviours)):
                directory = directories[i]
                # for this problem data
                problem_data = {}
                problem_data['human_behaviour'] = human_behaviours[i]
                problem_data['aq'] = 'LETHAM_UCB'
                problem_data['function'] = function.name
                problem_data['hum_mean'] = hum

                colors = ['tab:red','k','tab:green','tab:orange','tab:purple','tab:brown']
                # try:
                s_i = plot_regret(problem_data,axs,colors[i],directory,1000,b_w=False,unc=True,noise=None)
                # except:
                #     pass

            fig.suptitle('Bioprocess Control',x=0.5,y=0.1)

            fig,axs = format_plot(fig,axs,s_i,type='Bioprocess')
            if hum == True:
                plt.savefig('bo/plots/human/bioprocess_mean.pdf',dpi=400)
            else:
                plt.savefig('bo/plots/human/bioprocess.pdf',dpi=400)

def plot_reactor():
    directories = ['bo/reactor','bo/reactor_optimisation_human']
    colors = ['tab:red','k']
    human_behaviours = ['trusting','human']
    # directories = ['bo/reactor']
    # colors = ['tab:red']
    # human_behaviours = ['trusting']
    
    functions = [Reactor(4)]

    
    for hum in [True,False]:
        for function in functions:
            fig,axs = plt.subplots(1,2,figsize=(9,3))
            
            for i in range(len(human_behaviours)):
                directory = directories[i]
                # for this problem data
                problem_data = {}
                problem_data['human_behaviour'] = human_behaviours[i]
                problem_data['aq'] = 'LETHAM_UCB'
                problem_data['function'] = function.name
                problem_data['hum_mean'] = hum

                colors = ['tab:red','k','tab:green','tab:orange','tab:purple','tab:brown']
                # try:
                s_i = plot_regret(problem_data,axs,colors[i],directory,1000,b_w=False,unc=True,noise=None)
                # except:
                #     pass

            fig.suptitle('Reactor Geometry Optimization',x=0.5,y=0.1)

            fig,axs = format_plot(fig,axs,s_i,type='Reactor')
            if hum == True:
                plt.savefig('bo/plots/human/reactor_mean.pdf',dpi=400)
            else:
                plt.savefig('bo/plots/human/reactor.pdf',dpi=400)

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

# plot_reactor()

b_w = False
aq = 'EI'
# plot_noise_grid(aq,10000,b_w)
# plot_specific_grid(aq,0.025,10000,b_w)
# plot_alt_grid(10000,b_w)

aq = 'UCB'
noise = 0.0
d = 2
plot_rkhs(aq,noise,d,1000,b_w)
# for aq in ['EI','UCB']:
#     for noise in [0.0,0.05,0.1]:
#         plot_rkhs(aq,noise,2,1000,b_w)
#         plot_rkhs(aq,noise,3,1000,b_w)
#         plot_rkhs(aq,noise,5,1000,b_w)
        # plot_specific(aq,noise,10000,b_w)
# #plot_real(50,b_w)
