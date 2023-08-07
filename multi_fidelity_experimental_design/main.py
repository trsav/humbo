from multi_fidelity_experimental_design.utils import *
import os


def ed(
    f,
    data_path,
    x_bounds,
    z_bounds,
    problem_data,
    path,
    eval_error=True,
    printing=False,
):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    z_high = {}
    for k, v in z_bounds.items():
        z_high[k] = v[1]

    type = problem_data["type"]
    it_budget = problem_data["iterations"]
    sample_initial = problem_data["sample_initial"]
    gp_ms = problem_data["gp_ms"]
    ms_num = problem_data["ms_num"]

    j_bounds = x_bounds | z_bounds
    if type == "hf":
        s_bounds = x_bounds
    if type == "jf" or type == "mf":
        s_bounds = j_bounds

    samples = sample_bounds(s_bounds, sample_initial)

    data = {"data": []}
    for sample in samples:
        sample_dict = sample_to_dict(sample, s_bounds)
        s_eval = sample_dict.copy()
        if type == "hf":
            for zk, zv in z_high.items():
                s_eval[zk] = zv
        res = f(s_eval)
        run_info = {
            "id": res["id"],
            "inputs": sample_dict,
            "cost": res["cost"],
            "objective": res["objective"],
        }
        data["data"].append(run_info)
        save_json(data, data_path)

    data = read_json(data_path)
    data["problem_data"] = problem_data

    save_json(data, data_path)

    iteration = len(data["data"]) - 1

    while iteration < it_budget:
        start_time = time.time()
        data = read_json(data_path)
        inputs, outputs, cost = format_data(data)
        gp = build_gp_dict(*train_gp(inputs, outputs, gp_ms))
        c_gp = build_gp_dict(*train_gp(inputs, cost, gp_ms))
        if eval_error == True:
            n_test = 100
            x_test = sample_bounds(x_bounds, n_test)
            y_true = []
            y_test = []
            print("Evaluting model (never use this for an actual problem)")
            for x in tqdm(x_test):
                x_eval = {}
                x_keys = list(x_bounds.keys())
                for i in range(len(x_keys)):
                    x_eval[x_keys[i]] = x[i]
                for k, v in z_high.items():
                    x_eval[k] = v
                y_true.append(f(x_eval)["objective"])
                if type == "jf" or type == "mf":
                    x = np.concatenate((x, list(z_high.values())))
                    m, v = inference(gp, jnp.array([x]))
                    y_test.append(m)
            error = 0
            for i in range(n_test):
                error += (y_test[i] - y_true[i]) ** 2
            error /= n_test

        if printing == True:
            xk = list(x_bounds.keys())[0]
            x_sample = np.linspace(x_bounds[xk][0], x_bounds[xk][1], 200)
            mean = []
            cov = []
            for x in x_sample:
                conditioned_sample = jnp.array([[x]])
                if type == "jf" or type == "mf":
                    conditioned_sample = jnp.array(
                        [
                            jnp.concatenate(
                                (conditioned_sample, jnp.array([list(z_high.values())]))
                            )[:, 0]
                        ]
                    )
                mean_v, cov_v = inference(gp, conditioned_sample)
                mean.append(float(mean_v))
                cov.append(float(cov_v))

            y = []
            c = []
            x = np.linspace(x_bounds[xk][0], x_bounds[xk][1], 200)
            x_sample = {}
            for xi in x:
                x_sample[xk] = xi
                for k, v in z_high.items():
                    x_sample[k] = v
                e = f(x_sample)
                y.append(e["objective"])
                c.append(e["cost"])
            var = np.sqrt(np.array(cov))

            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
            ax.plot(x, y, c="k", lw=2, label="Highest Fidelity Function", alpha=0.5)
            if type == "hf":
                ax.scatter(inputs, outputs, c="k", s=20, lw=0, label="Data")
            else:
                for k in range(len(inputs)):
                    fid = np.float64(inputs[k, 1])
                    alpha = 0.1 + 0.9 * fid
                    size = 80 - 60 * fid
                    ax.scatter(
                        inputs[k, 0],
                        outputs[k],
                        c="k",
                        s=size,
                        alpha=alpha,
                        lw=0,
                        label="Data" if k == 0 else None,
                    )
            # remove top and right spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # print MSE in top left of plot
            ax.text(
                0.05,
                0.95,
                "Current MSE: " + str(np.round(error[0], 3)),
                transform=ax.transAxes,
            )
            ax.set_xlabel("$x$")
            ax.set_ylabel("$f(x)$")
            ax.plot(x, mean, c="k", ls="--", lw=2, label="Highest Fidelity Model")
            ax.fill_between(
                x,
                mean + 2 * var,
                mean - 2 * var,
                alpha=0.05,
                color="k",
                lw=0,
                label="95% Confidence",
            )
            # place legend below plot
            ax.legend(
                frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2
            )
            fig.tight_layout()

            plt.savefig(path + "/" + str(iteration + 1) + ".png", dpi=400)
        iteration += 1

        # optimising the aquisition of inputs, disregarding fidelity
        print("Optimising aquisition function")

        b_list = list(s_bounds.values())
        # sample and normalise initial guesses
        s_init = jnp.array(sample_bounds(s_bounds, ms_num))
        f_best = 1e20
        # define grad and value for acquisition (jax)
        if type == "hf" or type == "jf":
            f_aq = value_and_grad(exp_design_hf)
            args = gp
        else:
            f_aq = value_and_grad(exp_design_mf)
            x_b = jnp.array([b for b in list(x_bounds.values())])
            z_h = jnp.array(list(z_high.values()))
            args = (gp, c_gp, z_h, x_b)

        run_store = []

        for i in range(ms_num):
            s = s_init[i]
            res = minimize(
                f_aq,
                x0=s,
                args=args,
                method="SLSQP",
                bounds=b_list,
                jac=True,
                tol=1e-8,
                options={"disp": True},
            )
            aq_val = res.fun
            x = res.x
            run_store.append(aq_val)
            # if this is the best, then store solution
            if aq_val < f_best:
                f_best = aq_val
                x_opt = x

        mu_standard_obj, var_standard_obj = inference(gp, jnp.array([x_opt]))

        x_opt = list(x_opt)
        x_opt = [np.float64(xi) for xi in x_opt]
        print("unnormalised res:", x_opt)

        sample = sample_to_dict(x_opt, s_bounds)

        run_info = {
            "id": "running",
            "inputs": sample,
            "cost": "running",
            "objective": "running",
            "pred_obj_mean": np.float64(mu_standard_obj),
            "pred_obj_std": np.float64(np.sqrt(var_standard_obj)),
        }
        try:
            run_info["MSE"] = np.float64(error)
        except:
            print("No Error Calculation")
        data["data"].append(run_info)
        save_json(data, data_path)

        s_eval = sample.copy()
        if type == "hf":
            for zk, zv in z_high.items():
                s_eval[zk] = zv
        res = f(s_eval)

        for k, v in res.items():
            run_info[k] = v
        data["data"][-1] = run_info
        save_json(data, data_path)

        if printing == True and eval_error == True:
            MSE = [
                data["data"][i + sample_initial]["MSE"]
                for i in range(len(data["data"]) - sample_initial)
            ]
            if len(MSE) > 1:
                cum_cost = jnp.cumsum(
                    np.array(
                        [
                            data["data"][i + sample_initial]["cost"]
                            for i in range(len(data["data"]) - sample_initial)
                        ]
                    )
                )
            else:
                cum_cost = [data["data"][-1]["cost"]]
            fig, ax = plt.subplots(2, 1, figsize=(7, 4))
            ax[0].plot(cum_cost, MSE, c="k", lw=2)
            ax[1].plot(jnp.arange(len(MSE)), MSE, "k", lw=2)
            ax[0].set_xlabel("Time")
            ax[1].set_xlabel("Iteration")
            ax[0].set_ylabel("MSE")
            ax[1].set_ylabel("MSE")
            for ax_i in ax:
                ax_i.spines["top"].set_visible(False)
                ax_i.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(path + "/mse.png", dpi=400)
