import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from multi_fidelity_experimental_design.utils import *
from multi_fidelity_experimental_design.main import *
import uuid
import matplotlib

x_bounds = {}
x_bounds["x1"] = [2, 8]

z_bounds = {}
z_bounds["z1"] = [0, 1]


def eval(x: dict):
    x1 = x["x1"]
    z1 = x["z1"]
    f1 = np.sin(x1)  # low fidelity
    f2 = 2 * np.cos(x1) / (x1) + 0.5 * np.sin(3 * x1)  # high fidelity
    f = (1 - z1) * f1 + z1 * f2
    return {"obj": f, "cost": z1**2 + x1 * 0.15, "id": str(uuid.uuid4())}


def sample_toy(xb, z):
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


def plot_toy():
    cmap = matplotlib.cm.get_cmap("Spectral")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    xb = x_bounds["x1"]
    x, y, c = sample_toy(xb, z_bounds["z1"][1])
    ax[0].plot(
        x, y, c="k", lw=3, label=r"$f(x,z=1):= \frac{2\cos(x)}{x} + \frac{\sin(3x)}{2}$"
    )
    ax[1].plot(x, c, c="k", lw=3, label="$c(x,z=1)$")
    x, y, c = sample_toy(xb, z_bounds["z1"][0])
    ax[0].plot(x, y, c="tab:red", lw=3, label=r"$f(x,z=0):= \sin(x)$")
    ax[1].plot(x, c, c="tab:red", lw=3, label="$c(x,z=0)$")
    ax[1].legend(frameon=False, fontsize=8)
    n = 10
    rgbr = [214, 39, 40]
    rgbb = [0, 0, 0]
    for i in np.linspace(0, 1, n):
        col = np.array([i * rgbr[j] + (1 - i) * rgbb[j] for j in range(3)]) / 256
        x, y, c = sample_toy(xb, i * z_bounds["z1"][0] + (1 - i) * z_bounds["z1"][1])
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
    plt.savefig("toy/vis/vis.png", dpi=300)
    return 

# plot_toy()

ed_hf(
    eval,
    "toy/hf_only.json",
    x_bounds,
    {'z1':1},
    16,
    sample_initial=4,
    ms_num=16,
    gp_ms=4,
    printing=True,
    printing_path = 'toy/vis/'
)

