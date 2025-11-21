# config.py
"""
Global configuration file for the optimization project.
Modify values here only — all scripts will automatically use them.
"""

import os

CONFIG = {
    # optimiser -> choose "ga" or "fmincon"
    "optimiser":"fmincon",
    
    # Initial guess
    "x0": [75, 200],

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
    
    # Optimization settings
    "algorithm": "sqp",
    "display": "iter",
    "max_function_evaluation":100,    # default is 100*numberOfVariables
    "max_iterations":400,             # default is 400 
    "constraint_tolerance":1e-5,      # default -> 1e-6
    "elite_count":5,                  # default -> {ceil(0.05*PopulationSize)}
    "hybrid_fcn":"fmincon",           # Function that continues the optimization after ga terminates
    "max_generations":200,            # default is {100*numberOfVariables} for ga
    "pop_size" : 50,                  # default is {50} when numberOfVariables <= 5, {200} otherwise 
    "use_parallel":True,              # Compute fitness and nonlinear constraint functions in parallel
    
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
