import numpy as np
import nlopt
import mlflow
import dagshub
from config import CONFIG
import simulation


def run_nlopt():
    """
    fmincon-like optimization using NLopt to MAXIMIZE annual energy.

    Returns:
        x_opt_real (np.ndarray): best solution in real units
        best_energy (float): max annual energy
        opt (nlopt.opt): optimizer object
    """
    
    # database setup
    mlflow.set_tracking_uri("https://dagshub.com/aryanvj787/NYS-Design-Optimisation-using-PySAM.mlflow")
    dagshub.init(repo_owner='aryanvj787', repo_name='NYS-Design-Optimisation-using-PySAM', mlflow=True)


    # set experiment name
    mlflow.set_experiment("nlopt-fmincon-optimisation")
    
    # author tag
    mlflow.set_tag(
        "Author",CONFIG["author"]
    )

    # set run name here
    run_name = CONFIG["run_name"]

    with mlflow.start_run(run_name=run_name):
        
        var_names = CONFIG["overrides"]

        lb = np.array(CONFIG["lb"], dtype=float)
        ub = np.array(CONFIG["ub"], dtype=float)

        x0_real = np.array(CONFIG["x0_override"] if CONFIG["x0_override"] is not None else CONFIG["x0"], dtype=float)
        
        mlflow.log_params(
            {
                "lower bound":lb,
                "upper bound":ub,
                "x0_real":x0_real,
            }
        )

        dim = len(var_names)
        if dim == 0:
            raise ValueError("CONFIG['overrides'] is empty.")

        # ---------- scaling helpers ----------
        def to_unit(x_real):
            return (x_real - lb) / (ub - lb)

        def from_unit(x_unit):
            return lb + x_unit * (ub - lb)

        # choose search space
        if CONFIG["scale_to_unit"]:
            lb_opt = np.zeros(dim)
            ub_opt = np.ones(dim)
            x0_opt = to_unit(x0_real)
        else:
            lb_opt = lb.copy()
            ub_opt = ub.copy()
            x0_opt = x0_real.copy()

        # ---------- NLopt setup ----------
        algo = getattr(nlopt, CONFIG["nlopt_algorithm"])
        opt = nlopt.opt(algo, dim)

        opt.set_lower_bounds(lb_opt)
        opt.set_upper_bounds(ub_opt)

        opt.set_maxeval(CONFIG["maxeval"])
        opt.set_xtol_rel(CONFIG["xtol_rel"])
        opt.set_ftol_rel(CONFIG["ftol_rel"])

        # ---------- objective ----------
        def objective(x_opt, grad):
            # map to real space if scaling
            x_real = from_unit(x_opt) if CONFIG["scale_to_unit"] else np.array(x_opt, dtype=float)

            if CONFIG["round_integers"]:
                x_real = np.array([int(round(v)) for v in x_real], dtype=float)

            overrides = {var_names[i]: float(x_real[i]) for i in range(dim)}

            try:
                annual_energy = simulation.simulation.run_simulation(overrides)

                # guard against NaNs/Infs
                if annual_energy is None or not np.isfinite(annual_energy):
                    return -1e30

                return float(annual_energy)

            except Exception:
                # if sim crashes, give terrible fitness
                return -1e30

        opt.set_max_objective(objective)

        # ---------- run ----------
        x_opt_found = opt.optimize(x0_opt)
        best_energy = opt.last_optimum_value()

        x_opt_real = from_unit(x_opt_found) if CONFIG["scale_to_unit"] else np.array(x_opt_found, dtype=float)
        if CONFIG["round_integers"]:
            x_opt_real = np.array([int(round(v)) for v in x_opt_real], dtype=float)

        if CONFIG["verbose"]:
            print("\nNLopt finished.")
            print("Algorithm:", CONFIG["algorithm"])
            print("Status code:", opt.last_optimize_result())
            print("Best overrides:")
            x_dict = {}
            for name, val in zip(var_names, x_opt_real):
                rav = name + " optimal value"
                x_dict[rav] = float(val)
                # print(f"  {name}: {val:.6f}")
            print(x_dict)
            print("Max annual energy:", best_energy)
        mlflow.log_metrics(x_dict)
        mlflow.log_metric("Optimal function value",best_energy)

        return x_opt_real, best_energy, opt


# def multistart_nlopt(
#     n_starts=12,
#     algorithm="LD_SLSQP",
#     maxeval_per_start=150,
#     xtol_rel=1e-4,
#     ftol_rel=1e-4,
#     scale_to_unit=True,
#     round_integers=False,
#     seed=0,
#     verbose=True
# ):
#     """
#     Runs NLopt multiple times from random initial points and returns the best result.
#     Great for nonconvex landscapes.

#     Returns:
#         best_x (np.ndarray), best_energy (float)
#     """

#     lb = np.array(CONFIG["lb"], dtype=float)
#     ub = np.array(CONFIG["ub"], dtype=float)
#     dim = len(CONFIG["overrides"])

#     rng = np.random.default_rng(seed)

#     best_x = None
#     best_energy = -np.inf

#     for k in range(n_starts):
#         x0 = lb + rng.random(dim) * (ub - lb)

#         x_opt, energy, _ = run_nlopt(
#             algorithm=algorithm,
#             maxeval=maxeval_per_start,
#             xtol_rel=xtol_rel,
#             ftol_rel=ftol_rel,
#             scale_to_unit=scale_to_unit,
#             round_integers=round_integers,
#             x0_override=x0,
#             verbose=False
#         )

#         if energy > best_energy:
#             best_energy = energy
#             best_x = x_opt

#         if verbose:
#             print(f"Start {k+1}/{n_starts}: best_energy_so_far={best_energy:.6f}")

#     if verbose:
#         print("\nMulti-start NLopt best result:")
#         for name, val in zip(CONFIG["overrides"], best_x):
#             print(f"  {name}: {val:.6f}")
#         print("Max annual energy:", best_energy)

#     return best_x, best_energy
