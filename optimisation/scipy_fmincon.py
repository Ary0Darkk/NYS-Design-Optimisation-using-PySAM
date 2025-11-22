import numpy as np
from scipy.optimize import minimize

import config
import simulation


def run_scipy_minimise(method="trust-constr", maxiter=200, verbose=3):
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

    cfg = config.CONFIG
    var_names = cfg["overrides"]
    x0 = np.array(cfg["x0"], dtype=float)
    lb = np.array(cfg["lb"], dtype=float)
    ub = np.array(cfg["ub"], dtype=float)

    bounds = list(zip(lb, ub))

    # ---- objective: minimize negative annual energy ----
    def obj(x):
        overrides = {var_names[i]: float(x[i]) for i in range(len(var_names))}
        annual_energy = simulation.simulation.run_simulation(overrides)
        return -float(annual_energy)

    # ---- solve ----
    result = minimize(
        obj,
        x0=x0,
        method=method,
        bounds=bounds,
        constraints=[],  # add NonlinearConstraint/LinearConstraint here if needed
        options={"maxiter": maxiter, "verbose": verbose}
    )

    x_opt = result.x
    max_energy = -result.fun

    return x_opt, max_energy, result
