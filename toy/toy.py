import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from multi_fidelity_experimental_design.utils import *
from multi_fidelity_experimental_design.main import *
import uuid
import matplotlib
from multiprocessing import Pool

x_bounds = {}
x_bounds["x1"] = [2, 8]

z_bounds = {}
z_bounds["z1"] = [0, 1]





def sample_toy(xb, z,eval):
    x_sample = {}
    x_sample["z1"] = z
    y = []
    c = []
    x = np.linspace(xb[0], xb[1], 300)
    for xi in x:
        x_sample["x1"] = xi
        e = eval(x_sample)
        y.append(e["obj"])
        c.append(e["cost"])
    return x, y, c


def plot_toy(eval,path):
    cmap = matplotlib.cm.get_cmap("Spectral")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    xb = x_bounds["x1"]
    x, y, c = sample_toy(xb, z_bounds["z1"][1],eval)
    ax[0].plot(
        x, y, c="k", lw=3, label=r"$f(x,z=1):= \frac{2\cos(x)}{x} + \frac{\sin(3x)}{2}$"
    )
    ax[1].plot(x, c, c="k", lw=3, label="$c(x,z=1)$")
    x, y, c = sample_toy(xb, z_bounds["z1"][0],eval)
    ax[0].plot(x, y, c="tab:red", lw=3, label=r"$f(x,z=0):= \sin(x)$")
    ax[1].plot(x, c, c="tab:red", lw=3, label="$c(x,z=0)$")
    ax[1].legend(frameon=False, fontsize=8)
    n = 10
    rgbr = [214, 39, 40]
    rgbb = [0, 0, 0]
    for i in np.linspace(0, 1, n):
        col = np.array([i * rgbr[j] + (1 - i) * rgbb[j] for j in range(3)]) / 256
        x, y, c = sample_toy(xb, i * z_bounds["z1"][0] + (1 - i) * z_bounds["z1"][1],eval)
        ax[0].plot(x, y, c=tuple(col), lw=3, alpha=0.2)
        ax[1].plot(x, c, c=tuple(col), lw=3, alpha=0.2)

    ax[0].set_xlabel("x", fontsize=14)
    ax[0].set_ylabel("f(x)", fontsize=14)
    ax[1].set_xlabel("x", fontsize=14)
    ax[1].set_ylabel("$c(x)$", fontsize=14)

    for axs in ax:
        axs.spines["top"].set_visible(False)
        axs.spines["right"].set_visible(False)
        axs.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    plt.savefig(path+"vis.png", dpi=300)
    return 

# plot_toy()

class Problem:
    def __init__(self,alpha,cost_ratio,cb):
        self.alpha = alpha
        self.cost_ratio = cost_ratio
        self.cb = cb

    def eval(self,x: dict):
        x1 = x["x1"]
        z1 = x["z1"]
        f_low = np.sin(x1)  # low fidelity
        f_high_1 = 2 * np.cos(x1) / (x1) + 0.5 * np.sin(3 * x1)  # high fidelity
        f_high_2 = f_low - 1
        f_high = self.alpha * f_high_1 + (1-self.alpha) * f_high_2
        c_low = 1
        c_high = self.cost_ratio
        if self.cb == 'linear':
            cost = (c_high-c_low) * z1 + c_low
        if self.cb == 'exp':
            # lowest cost is c_low then rising exponentially to c_high
            cost = (c_high-c_low) * z1 ** 2 + c_low
        f = (1 - z1) * f_low + z1 * f_high
        return {"obj": f, "cost": cost, "id": str(uuid.uuid4())}




n = 36
alpha = np.linspace(0,1,n)
cost_ratio = np.geomspace(1.1,100,n)
# shuffle cost_ratio
np.random.shuffle(cost_ratio)


# create meshgrid from alpha and cost_ratio
alpha, cost_ratio = np.meshgrid(alpha, cost_ratio)
alpha = alpha.flatten()
cost_ratio = cost_ratio.flatten()
# repeat alpha and cost_ratio for hf and jf
types = ['hf' for i in range(len(alpha))] + ['jf' for i in range(len(alpha))]
print(types)
alpha = np.concatenate((alpha,alpha))
cost_ratio = np.concatenate((cost_ratio,cost_ratio))

cost_behaviours = ['linear' for i in range(len(alpha))] + ['exp' for i in range(len(alpha))]
alpha = np.concatenate((alpha,alpha))
cost_ratio = np.concatenate((cost_ratio,cost_ratio))
types = np.concatenate((types,types))

def task(args):
    i, (alpha_v, cost_ratio_v,type,cost_behaviour) = args

    eval = Problem(alpha_v,cost_ratio_v,cost_behaviour).eval

    a_string = 'a_'+str(np.round(alpha_v,3)) + '_cr_' + str(np.round(cost_ratio_v)) + '_cb_' + cost_behaviour + '_sol_'+ type + '/'
    path = 'toy/res_'+a_string+'/'
    try: 
        os.mkdir(path)
    except:
        pass
    #plot_toy(eval,path)
    ed(
        eval,
        path+"res.json",
        x_bounds,
        {'z1':[0,1]},
        16,
        sample_initial=4,
        ms_num=16,
        gp_ms=4,
        printing=False,
        printing_path = path,
        type=type
    )

    return

def main():
    with Pool(processes=18) as pool:
        pool.map(task, enumerate(zip(alpha, cost_ratio,types, cost_behaviours)))

if __name__ == "__main__":
    main()