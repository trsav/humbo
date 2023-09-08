from utils import * 
from tqdm import tqdm
from matplotlib import rc
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

def plot_regret(problem_data,axs,c):
    directory = 'bo'
    files = os.listdir(directory)
    problem_data_list = []
    for i in tqdm(range(len(files))):
        if '.' not in files[i] and '_' not in files[i]:
            results = directory+'/'+files[i] + '/res.json'
            # open json
            with open(results, "r") as f:
                data = json.load(f)
            problem_data_list.append(data['problem_data'])

    # create dataframe from list of dictionaries 
    df = pd.DataFrame(problem_data_list)

    df = df.loc[(df['sample_initial'] == problem_data['sample_initial']) & (df['gp_ms'] == problem_data['gp_ms']) & (df['alternatives'] == problem_data['alternatives']) & (df['NSGA_iters'] == problem_data['NSGA_iters']) & (df['max_iterations'] == problem_data['max_iterations']) & (df['human_behaviour'] == problem_data['human_behaviour']) & (df['acquisition_function'] == problem_data['acquisition_function'])]
    file_names = df['file_name'].values
    regret_list = []
    obj_list = []
    f_opt_list = []
    for file in file_names:
        data_full = read_json(file + '/res.json')
        data = data_full['data']
        f_opt = data_full['problem_data']['f_opt']
        obj = [d['objective'] for d in data]
        f_opt_list.append(f_opt)
        obj_list.append(obj)

    init = data_full['problem_data']['sample_initial']
    full_it = problem_data['max_iterations']

    average_regret_list = []
    regret_list = []
    for obj,i in zip(obj_list,range(len(obj_list))):
        if len(obj) != full_it:
            obj += [obj[-1]]*(full_it-len(obj))
        it = len(obj)
        regret = [f_opt_list[i] - np.max(obj[:t]) for t in range(1,it+1)]
        regret_list.append(regret)
        cumulative_regret = [f_opt_list[i] - np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret = [f_opt_list[i] - (1/t) * np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret_list.append(average_regret)
    average_regret = np.mean(np.array(average_regret_list),axis=0)
    average_regret_std = np.std(np.array(average_regret_list),axis=0)
    label = problem_data['human_behaviour']
    if label.__class__ != float:
        label = label[0].upper() + label[1:]
    else:
        label = '$p($Best$)=$'+str(label) 
    # label = '$\mathbb{E}[$'+label+'$]$'
    # captialise first letter 
    x = np.arange(init,len(average_regret))

    axs[1].plot(x,average_regret[init:],c=c,lw=1.5,label=label)
    axs[1].fill_between(x,average_regret[init:]-average_regret_std[init:],average_regret[init:]+average_regret_std[init:],alpha=0.1,color=c)


    ax = axs[0]
    regret_list = np.array(regret_list)
    mean_instantaneous_regret = np.mean(regret_list,axis=0)
    std_instantaneous_regret = np.std(regret_list,axis=0)
    x = np.arange(init,len(mean_instantaneous_regret))
    ax.plot(x,mean_instantaneous_regret[init:],c=c,lw=1.5,label=label)
    ax.fill_between(x,mean_instantaneous_regret[init:]-std_instantaneous_regret[init:],mean_instantaneous_regret[init:]+std_instantaneous_regret[init:],alpha=0.1,color=c)

    return 

colors = ['tab:red','tab:blue','tab:green','tab:orange','tab:purple','tab:brown']
human_behaviours = ['expert','adversarial','trusting',0.25,0.5,0.75]

for aq in ['UCB','EI']
    fig,axs = plt.subplots(1,2,figsize=(10,4))

    for i in range(len(human_behaviours)):
        aq = 'UCB'
        problem_data = {}
        problem_data["sample_initial"] = 4
        problem_data["gp_ms"] = 8
        problem_data["alternatives"] = 3
        problem_data["NSGA_iters"] = 50
        problem_data["plotting"] = True
        problem_data['max_iterations'] = 75
        problem_data['lengthscale'] = 0.5
        # at a given human behaviour
        problem_data['human_behaviour'] = human_behaviours[i]
        problem_data['acquisition_function'] = aq

        plot_regret(problem_data,axs,colors[i])
    fs = 12
    axs[0].set_ylabel(r"Simple Regret, $r_\tau$",fontsize=fs)
    for ax in axs:
        ax.grid(True,alpha=0.5)
        x_start = problem_data['sample_initial']
        max_y = ax.get_ylim()[1]
        min_y = ax.get_ylim()[0]
        ax.plot([x_start,x_start],[min_y,max_y],c='k',ls='--',lw=1,alpha=0.5)
    axs[0].set_xlabel(r"Iterations, $\tau$",fontsize=fs)
    axs[1].set_xlabel(r"Iterations, $\tau$",fontsize=fs)
    axs[1].set_ylabel(r"Average Regret, ${R_\tau}/{\tau}$",fontsize=fs)

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=6,frameon=False)

    l = problem_data['lengthscale']

    fig.suptitle(r'Regret expectation over 50 functions, $f \sim \mathcal{GP}(\mu \equiv 0, K_M (d,\nu = '+str(l)+'))$, '+str(problem_data['alternatives'])+' alternate choices, $\mathcal{U}(x)=$'+str(aq),fontsize=int(fs))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    plt.savefig('bo/overall_regret.pdf')
