import numpy as np
from scipy.optimize import minimize
import mlflow
import dagshub
from config import CONFIG
import simulation


def run_scipy_minimise():
    """
    fmincon-like constrained optimization using SciPy.
    Maximizes annual energy by minimizing negative energy.

    Parameters
    ----------
    method : str
        "trust-constr" (most fmincon-like), or "SLSQP".
    maxiter : int
        Maximum iterations.
    verbose : int
        trust-constr verbosity: 0,1,2,3. For SLSQP, use 0/1.

    Returns
    -------
    x_opt : np.ndarray
        Best decision variables (same order as CONFIG["overrides"])
    max_energy : float
        Maximum annual energy
    result : OptimizeResult
        Full SciPy result object
    """
    # database setup
    mlflow.set_tracking_uri("https://dagshub.com/aryanvj787/NYS-Design-Optimisation-using-PySAM.mlflow")
    dagshub.init(repo_owner='aryanvj787', repo_name='NYS-Design-Optimisation-using-PySAM', mlflow=True)

    # set experiment name
    mlflow.set_experiment("scipy-minimise-ga-optimisation")

    # set run name here
    run_name = None

    with mlflow.start_run(run_name=run_name):
        
        print('Running scipy minimise optimiser...')
        
        var_names = CONFIG["overrides"]
        x0 = np.array(CONFIG["x0"], dtype=float)
        lb = np.array(CONFIG["lb"], dtype=float)
        ub = np.array(CONFIG["ub"], dtype=float)
        
        mlflow.log_params(
            {
                "Initial guess":x0,
                "Lower Bound":lb,
                "Upper Bound":ub,
            }
        )

        bounds = list(zip(lb, ub))
        
        mlflow.log_param("Bounds",bounds)

        # ---- objective: minimize negative annual energy ----
        def obj(x):
            overrides = {var_names[i]: float(x[i]) for i in range(len(var_names))}
            annual_energy = simulation.simulation.run_simulation(overrides)
            return -float(annual_energy)
        options = {
            "maxiter": CONFIG["maxiter"], 
            "verbose": CONFIG["verbose"],
            }
        mlflow.log_params(options)
        # ---- solve ----
        result = minimize(
            obj,
            x0=x0,
            method=CONFIG["method"],
            bounds=bounds,
            constraints=[],  # add NonlinearConstraint/LinearConstraint here if needed
            options=options
        )

        x_opt = result.x
        max_energy = -result.fun
        
        x_dict={}
        for var,value in zip(CONFIG["overrides"],x_opt):
            rav = var + " optimal value"
            x_dict[rav] = float(value)
        
        mlflow.log_metrics(x_dict)
        mlflow.log_metric("Optimal function Value",max_energy)

        return x_opt, max_energy, result
