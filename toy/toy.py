import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from multi_fidelity_experimental_design.utils import *
from multi_fidelity_experimental_design.main import *
import uuid
from multiprocessing import Pool, cpu_count


class Problem:
    def __init__(self, alpha, cost_ratio, cb):
        self.alpha = alpha
        self.cost_ratio = cost_ratio
        self.cb = cb

    def eval(self, x: dict):
        x1 = x["x1"]
        z1 = x["z1"]
        f_low = np.sin(x1)  # low fidelity
        f_high_1 = 2 * np.cos(x1) / (x1) + 0.5 * np.sin(3 * x1)  # high fidelity
        f_high_2 = f_low - 1
        f_high = self.alpha * f_high_1 + (1 - self.alpha) * f_high_2
        c_low = 1
        c_high = self.cost_ratio
        if self.cb == "linear":
            cost = (c_high - c_low) * z1 + c_low
        if self.cb == "exp":
            # lowest cost is c_low then rising exponentially to c_high
            cost = (c_high - c_low) * z1**2 + c_low
        f = (1 - z1) * f_low + z1 * f_high
        return {"objective": f, "cost": cost, "id": str(uuid.uuid4())}


x_bounds = {}
x_bounds["x1"] = [2, 8]

z_bounds = {}
z_bounds["z1"] = [0, 1]

# p_string = str(uuid.uuid4())
# problem_data = {}
# problem_data["alpha"] = 0.5
# problem_data["cost_ratio"] = 10
# problem_data["cost_behaviour"] = "exp"
# problem_data["type"] = "mf"
# problem_data["sample_initial"] = 4
# problem_data["ms_num"] = 8
# problem_data["gp_ms"] = 4
# problem_data["iterations"] = 15
# eval = Problem(
#     problem_data["alpha"], problem_data["cost_ratio"], problem_data["cost_behaviour"]
# ).eval


# path = "toy/" + p_string + "/"
# try:
#     os.mkdir(path)
# except:
#     pass
# plot_toy(eval, path, x_bounds,z_bounds)
# ed(
#     eval,
#     path + "res.json",
#     x_bounds,
#     z_bounds,
#     problem_data,
#     path=path,
#     printing=True,
#     eval_error=True,
# )

n = 36
alpha = np.linspace(0, 1, n)
cost_ratio = np.geomspace(1.1, 100, n)
# shuffle cost_ratio
np.random.shuffle(cost_ratio)


# create meshgrid from alpha and cost_ratio
alpha, cost_ratio = np.meshgrid(alpha, cost_ratio)
alpha = alpha.flatten()
cost_ratio = cost_ratio.flatten()
# repeat alpha and cost_ratio for hf and jf
types = (
    ["hf" for i in range(len(alpha))]
    + ["jf" for i in range(len(alpha))]
    + ["mf" for i in range(len(alpha))]
)
alpha = np.concatenate((alpha, alpha, alpha))
cost_ratio = np.concatenate((cost_ratio, cost_ratio, cost_ratio))

cost_behaviours = ["linear" for i in range(len(alpha))] + [
    "exp" for i in range(len(alpha))
]
alpha = np.concatenate((alpha, alpha))
cost_ratio = np.concatenate((cost_ratio, cost_ratio))
types = np.concatenate((types, types))


def task(args):
    i, (alpha_v, cost_ratio_v, type, cost_behaviour) = args

    eval = Problem(alpha_v, cost_ratio_v, cost_behaviour).eval

    p_string = str(uuid.uuid4())
    path = "toy/" + p_string + "/"
    try:
        os.mkdir(path)
    except:
        pass

    # plot_toy(eval, path, x_bounds, z_bounds)
    problem_data = {}
    problem_data["alpha"] = alpha_v
    problem_data["cost_ratio"] = cost_ratio_v
    problem_data["cost_behaviour"] = cost_behaviour
    problem_data["type"] = type
    problem_data["sample_initial"] = 4
    problem_data["ms_num"] = 16
    problem_data["gp_ms"] = 4
    problem_data["iterations"] = 16
    eval = Problem(
        problem_data["alpha"],
        problem_data["cost_ratio"],
        problem_data["cost_behaviour"],
    ).eval

    ed(
        eval,
        path + "res.json",
        x_bounds,
        z_bounds,
        problem_data,
        path=path,
        printing=False,
        eval_error=True,
    )

    return

def main():
    with Pool(processes=32) as pool:
        pool.map(task, enumerate(zip(alpha, cost_ratio,types, cost_behaviours)))

if __name__ == "__main__":
    main()
