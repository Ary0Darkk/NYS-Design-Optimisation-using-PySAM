from optimisation.ga_optimiser import run_ga_optimisation
from optimisation.fmincon_optimiser import run_fmincon_optimisation
from optimisation.pygad_ga_optimiser import run_pyga_optimisation
from optimisation.scipy_fmincon import run_scipy_minimise
from optimisation.nlopt_fmincon import run_nlopt
from optimisation.deap_ga_optimiser import run_deap_ga_optimisation
from config import CONFIG

import argparse
import glob
import os  


def load_repro_config(path):
    """Load config (config.py-style dict)"""
    if path.endswith(".py"):
        # risky but works: execute file and capture CONFIG
        ns = {}
        exec(open(path).read(), ns)
        return ns.get("CONFIG", None)
    else:
        raise ValueError("Unsupported repro config format")

def find_latest_downloaded_config():
    files = sorted(
        glob.glob("downloaded_run_artifacts/*.py"),
        key=os.path.getmtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError("No downloaded config found in downloaded_run_artifacts/")
    return files[0]


def main():
    
    # simulation
    # sim_output = run_simulation()
    # print(f'Annual Energy : {sim_output["annual_energy"]}')
    
    # FIXME : try and catch is not working as expected,
    # look at keyboard interupt working
    try:
        # optimisation
        if CONFIG["optimiser"] == "fmincon":
            x_opt, f_val = run_fmincon_optimisation()
        elif CONFIG["optimiser"] == "ga":
            x_opt, f_val = run_ga_optimisation()
        elif CONFIG["optimiser"] == "pygad_ga":
            x_opt, f_val, _ = run_pyga_optimisation()
        elif CONFIG["optimiser"] == "nlopt":
            x_opt, f_val, _ = run_nlopt()
        elif CONFIG["optimiser"] == "scipy_min":
            x_opt, f_val, _ = run_scipy_minimise()
        elif CONFIG["optimiser"] == "scipy_min":
            x_opt, f_val, _ = run_scipy_minimise()
        elif CONFIG["optimiser"] == "deap_ga":
            x_opt, f_val, _ = run_deap_ga_optimisation()
        # disp optimal values
        print(f'x_opt : {x_opt} \nf_val : {f_val}')
    except KeyboardInterrupt:
        print('\n\nOptimization interrupted by user. Stopping...\n')
    except Exception as e:
        print('Unexpected error :',e)
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--repro", nargs="?", const="AUTO", help="Reproduce using downloaded config. Use AUTO for latest.")
    args = parser.parse_args()

    # override CONFIG if repro mode is used
    if args.repro:
        if args.repro == "AUTO":
            repro_file = find_latest_downloaded_config()
            print(f"[REPRO] Auto-loading latest downloaded config: {repro_file}")
        else:
            repro_file = args.repro
            print(f"[REPRO] Loading specified config: {repro_file}")

        repro_cfg = load_repro_config(repro_file)
        if repro_cfg is None:
            raise RuntimeError("Failed to load CONFIG from repro file.")

        print("[REPRO] Overriding CONFIG with loaded values...")
        CONFIG.update(repro_cfg if "CONFIG" not in repro_cfg else repro_cfg["CONFIG"])

    # proceed with normal run using modified CONFIG
    main()