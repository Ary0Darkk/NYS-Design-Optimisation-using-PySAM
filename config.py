# config.py
"""
Global configuration file for the optimization project.
Modify values here only — all scripts will automatically use them.
"""

import os

CONFIG = {
    # optimiser -> choose "ga" or "fmincon"
    "optimiser":"ga",
    
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
