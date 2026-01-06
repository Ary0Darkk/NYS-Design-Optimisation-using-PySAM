# config.py
"""
Global configuration file for the optimization project.
Modify values here only — all scripts will automatically use them.
"""

# FIXME: Need to find variable name for "Number of SCA per loop"
import os

CONFIG = {
    # author name -> whenever run it write your name
    "author": "Aryan",
    "run_name": None,
    "demand_file_path": "electricity_data/Yearly_Demand_Profile_state_mahrastra_and_manipur.xlsx",
    "show_demand_plot": False,
    "show_price_plot": False,
    # optimiser -> choose "ga" or "fmincon"
    "optimiser": "rl_optim",  # Initial guess
    "route": "design",  # des-operational
    "resume_from_checkpoint": False,
    "force_update": False,
    "storage_block": "local-file-system/gdrive-storage",
    "num_cores": 1000,
    # overrides
    "design": {
        "overrides": [
            # "specified_total_aperture",  # total aperture area
            "Row_Distance",  # row spacing
            "ColperSCA",  # num of modules per SCA
            "W_aperture",  # width of SCA
            "L_SCA",  # length of collector assembly
            # "nSCA",  # number of SCA per loop
        ],
        # Bounds
        "lb": [5, 2, 1, 40],
        "ub": [20, 10, 10, 150],
        "types": [
            float,
            int,
            float,
            float,
        ],
    },
    "operational": {
        "overrides": [
            # "m_dot_htfmin",  # min mass-flow rate
            # "m_dot_htfmax",  # max mass-flow rate
            "T_startup",  # startup temp
            "T_shutdown",  # shutdown temp
        ],
        # Bounds
        "lb": [30, 30],
        "ub": [100, 135],
        "types": [float, float],
    },
    # deap-ga optimisation settings
    "random_seed": 21,
    "tournament_size": 2,
    "mutation_num_genes": 5,
    "sol_per_pop": 4,
    "num_generations": 8,
    "cxpb": 0.8,
    "mutpb": 0.2,
    "indpb": 0.2,
    "verbose": True,
    # rl-based optimisation settings
    "rl_max_steps": 5,
    "rl_eval_steps": 10,
    "rl_lr": 3e-2,
    "rl_checkpoint_freq": 20,
    "rl_timesteps": 20,
    # nlopt-fmincon settings
    "nlopt_algorithm": "LD_SLSQP",
    "maxeval": 3,
    "xtol_rel": 1e-4,
    "ftol_rel": 1e-4,
    "scale_to_unit": True,
    "round_integers": False,
    "x0_override": None,
    # "verbose":True,
    # scipy-ga settings
    "method": "trust-constr",
    "maxiter": 2,
    # "verbose":3,
    # pygad-ga settings
    # "num_generations":1,
    # "sol_per_pop":4,
    "num_parents_mating": 1,
    # "mutation_num_genes":1,
    # "random_seed":None,
    # "verbose":True,
    # Optimization settings
    "algorithm": "sqp",
    "display": "iter-detailed",
    "OptimalityTolerance": 1e-6,
    "max_function_evaluation": 1,  # default is 100*numberOfVariables
    "max_iterations": 1,  # default is 400
    "constraint_tolerance": 1e-1,  # default -> 1e-6
    "elite_count": 1,  # default -> {ceil(0.05*PopulationSize)}
    "hybrid_fcn": None,  # Function that continues the optimization after ga terminates
    "max_generations": 2,  # default is {100*numberOfVariables} for ga
    "pop_size": 5,  # default is {50} when numberOfVariables <= 5, {200} otherwise
    "use_parallel": False,  # Compute fitness and nonlinear constraint functions in parallel
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
