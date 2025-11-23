# config.py
"""
Global configuration file for the optimization project.
Modify values here only — all scripts will automatically use them.
"""

import os

CONFIG = {
    # optimiser -> choose "ga" or "fmincon"
    "optimiser":"nlopt",
    
    # Initial guess
    "x0": [67, 200],

    # Bounds
    "lb": [50, 150],
    "ub": [100, 250],
    
    # overrides
    "overrides": [
        "T_startup",
        "T_shutdown"
        # "I_bn_des",
        # "TrackingError"
    ],
    
    # nlopt-fmincon settings
    "nlopt_algorithm":"LD_SLSQP",
    "maxeval":2,
    "xtol_rel":1e-4,
    "ftol_rel":1e-4,
    "scale_to_unit":True,
    "round_integers":False,
    "x0_override":None,
    "verbose":True,
    
    # scipy-ga settings
    "method":"trust-constr",
    "maxiter":2,
    "verbose":3,
    # pygad-ga settings
    "num_generations":1,
    "sol_per_pop":4,
    "num_parents_mating":1,
    "mutation_num_genes":1,
    "random_seed":None,
    "verbose":True,
    # Optimization settings
    "algorithm": "sqp",
    "display": "iter",
    "max_function_evaluation":1,    # default is 100*numberOfVariables
    "max_iterations":1,             # default is 400 
    "constraint_tolerance":1e-1,      # default -> 1e-6
    "elite_count": 1,                  # default -> {ceil(0.05*PopulationSize)}
    "hybrid_fcn":None,           # Function that continues the optimization after ga terminates
    "max_generations":2,            # default is {100*numberOfVariables} for ga
    "pop_size" : 5,                  # default is {50} when numberOfVariables <= 5, {200} otherwise 
    "use_parallel":False,              # Compute fitness and nonlinear constraint functions in parallel
    
    # Function names INSIDE the MATLAB file
    "objective_name": "obj_function",
    "constraint_name": "constraints",

    # Folder where MATLAB file lives
    "matlab_folder": os.path.join(os.path.dirname(__file__), "objective_functions"),

    
    # Input JSON File (PYSAM default input)
    "json_file": "simulation/Kanniyan_Project.json",
    
    # model we are using from PySAM
    "model":"PhysicalTroughNone",

}
