from utils import * 
from tqdm import tqdm
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

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

    df = df.loc[(df['sample_initial'] == problem_data['sample_initial']) & (df['gp_ms'] == problem_data['gp_ms']) & (df['alternatives'] == problem_data['alternatives']) & (df['NSGA_iters'] == problem_data['NSGA_iters']) & (df['regret_tolerance'] == problem_data['regret_tolerance']) & (df['max_iterations'] == problem_data['max_iterations']) & (df['human_behaviour'] == problem_data['human_behaviour']) & (df['acquisition_function'] == problem_data['acquisition_function'])]
    file_names = df['file_name'].values
    regret_list = []
    obj_list = []
    for file in file_names:
        data_full = read_json(file + '/res.json')
        data = data_full['data']
        f_opt = data_full['problem_data']['f_opt']
        regret = [d['regret'] for d in data]
        obj = [d['objective'] for d in data]
        while len(regret) != problem_data['max_iterations']:
            regret.append(problem_data['regret_tolerance'])
        regret_list.append(regret)
        obj_list.append(obj)
    regret_list = np.array(regret_list)

    init = data_full['problem_data']['sample_initial']
    full_it = problem_data['max_iterations']

    average_regret_list = []
    for obj,i in zip(obj_list,range(len(obj_list))):
        if len(obj) != full_it:
            obj += [obj[-1]]*(full_it-len(obj))
        it = len(obj)
        cumulative_regret = [f_opt - np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret = [f_opt - (1/t) * np.sum(obj[:t]) for t in range(1,it+1)]
        average_regret_list.append(average_regret)
    average_regret = np.mean(np.array(average_regret_list),axis=0)
    label = problem_data['human_behaviour']
    label = label[0].upper() + label[1:]
    # label = '$\mathbb{E}[$'+label+'$]$'
    # captialise first letter 
    axs[1].plot(np.arange(init,len(average_regret)),average_regret[init:],c=c,lw=1.5,label=label)

    ax = axs[0]
    regret_list = np.array(regret_list)
    mean_instantaneous_regret = np.mean(regret_list,axis=0)
    ax.plot(np.arange(init,len(mean_instantaneous_regret)),mean_instantaneous_regret[init:],c=c,lw=1.5,label=label)

    return 

colors = ['tab:red','tab:blue','tab:green','k']
human_behaviours = ['expert','idiot','trusting','random']
fig,axs = plt.subplots(1,2,figsize=(10,4))

for i in range(len(human_behaviours)):
    aq = 'UCB'
    problem_data = {}
    problem_data["sample_initial"] = 4
    problem_data["gp_ms"] = 8
    problem_data["alternatives"] = 4
    problem_data["NSGA_iters"] = 50
    problem_data["plotting"] = True
    problem_data['regret_tolerance'] = 0.0001
    problem_data['max_iterations'] = 50
    problem_data['lengthscale'] = 0.5
    # at a given human behaviour
    problem_data['human_behaviour'] = human_behaviours[i]
    problem_data['acquisition_function'] = aq

    plot_regret(problem_data,axs,colors[i])
fs = 12
axs[0].set_ylabel(r"Instantaneous Regret, $r_\tau$",fontsize=fs)
axs[0].set_xlabel(r"Iterations, $\tau$",fontsize=fs)
axs[1].set_xlabel(r"Iterations, $\tau$",fontsize=fs)
axs[1].set_ylabel(r"Average Regret, ${R_\tau}/{\tau}$",fontsize=fs)

axs[0].legend(frameon=False)
axs[1].legend(frameon=False)
fig.tight_layout()
plt.savefig('bo/overall_regret.pdf')