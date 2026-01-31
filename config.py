# config.py
"""
Global configuration file for the optimization project.
Modify values here only — all scripts will automatically use them.
"""

# NOTE: Need to find variable name for "Number of SCA per loop", this variable is found on outputs section in pysam
import os
from datetime import datetime

CONFIG = {
    # author name -> whenever run it write your name
    "author": "Aryan",
    "run_name": None,
    "session_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "demand_file_path": "electricity_data/Yearly_Demand_Profile_state_mahrastra_and_manipur.xlsx",
    "show_demand_plot": True,
    "show_price_plot": True,
    "is_tuning": False,
    # ---- optimiser -> choose "deap_ga" or "rl_optim"-------
    "optimiser": "deap_ga",  # Initial guess
    "route": "design_operational",  # "design" or "operational" or "design_operational"
    "resume_from_checkpoint": False,
    "refresh_cache": True,
    # "storage_block": "local-file-system/local-storage",
    "num_cores": 4,
    "penalty": -1e13,
    "USER_DEFINED_DAYS": {
        "winter": [(1, 5), (1, 18), (2, 3), (2, 20), (12, 5), (12, 18), (12, 28)],
        "summer": [(3, 10), (3, 25), (4, 8), (4, 22), (5, 5), (5, 18), (5, 30)],
        "monsoon": [(6, 10), (6, 25), (7, 8), (7, 22), (8, 5), (8, 18), (8, 30)],
        "post": [(9, 10), (9, 25), (10, 8), (10, 22), (11, 5), (11, 18), (11, 30)],
    },
    # ------ overrides --------------------------------------
    "design": {
        "overrides": [
            "specified_total_aperture",  # total aperture area
            "Row_Distance",  # row spacing
            "ColperSCA",  # num of modules per SCA
            "W_aperture",  # width of SCA
            "L_SCA",  # length of collector assembly
        ],
        # Bounds
        "lb": [7000, 2, 2, 1, 40],
        "ub": [9000, 20, 10, 10, 150],
        "types": [
            float,
            float,
            int,
            float,
            float,
        ],
    },
    "operational": {
        "overrides": [
            "m_dot",  # mass-flow rate
            "T_startup",  # startup temp
            "T_shutdown",  # shutdown temp
        ],
        # Bounds
        "lb": [2, 30, 30],
        "ub": [12, 100, 135],
        "types": [int, float, float],
    },
    # -----deap-ga optimisation settings--------------------
    "checkpoint_interval": 1,
    "random_seed": 44,
    "tournament_size": 7,
    "pop_size": 10,  # polulation size
    "hall_of_fame_size": 5,  # elites we preserve from each gen
    "num_generations": 2,
    "cxpb": 0.8,  # prob of mating an ind
    "mutpb": 0.6,  # prob of mutating an ind
    "indpb": 0.4,  # decides how much a chosen individual changes,generally 1/num of variables
    "verbose": True,
    # ------rl-based optimisation settings-------------------
    "rl_max_steps": 3,
    "rl_eval_steps": 2,
    "rl_timesteps": 10,
    "rl_epochs": 10,
    "rl_gamma": 0.99,
    "rl_lr": 3e-1,
    "rl_checkpoint_freq": 10,
    "rl_timesteps": 10,
    "rl_batch_size": 10,
    "rl_ent_coef": 0,
    # ------- other optimiser (not available) ------------------------------------------------
    # nlopt-fmincon settings
    # "nlopt_algorithm": "LD_SLSQP",
    # "maxeval": 3,
    # "xtol_rel": 1e-4,
    # "ftol_rel": 1e-4,
    # "scale_to_unit": True,
    # "round_integers": False,
    # "x0_override": None,
    # "verbose":True,
    # scipy-ga settings
    # "method": "trust-constr",
    # "maxiter": 2,
    # "verbose":3,
    # pygad-ga settings
    # "num_generations":1,
    # "sol_per_pop":4,
    # "num_parents_mating": 1,
    # "mutation_num_genes":1,
    # "random_seed":None,
    # "verbose":True,
    # Optimization settings
    # "algorithm": "sqp",
    # "display": "iter-detailed",
    # "OptimalityTolerance": 1e-6,
    # "max_function_evaluation": 1,  # default is 100*numberOfVariables
    # "max_iterations": 1,  # default is 400
    # "constraint_tolerance": 1e-1,  # default -> 1e-6
    # "elite_count": 1,  # default -> {ceil(0.05*PopulationSize)}
    # "hybrid_fcn": None,  # Function that continues the optimization after ga terminates
    # "max_generations": 2,  # default is {100*numberOfVariables} for ga
    # "pop_size": 5,  # default is {50} when numberOfVariables <= 5, {200} otherwise
    # "use_parallel": False,  # Compute fitness and nonlinear constraint functions in parallel
    # -------- SYSTEM SETTINGS-------------------------------
    # Function names INSIDE the MATLAB file
    "objective_name": "obj_function",
    "constraint_name": "constraints",
    # Folder where MATLAB file lives
    "matlab_folder": os.path.join(os.path.dirname(__file__), "objective_functions"),
    # Input JSON File (PYSAM default input)
    "json_file": "simulation/Gurgaon_plant.json",
    # model we are using from PySAM
    "model": "PhysicalTroughNone",
}
