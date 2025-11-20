# config.py
"""
Global configuration file for the optimization project.
Modify values here only — all scripts will automatically use them.
"""

import os

CONFIG = {
    # Initial guess
    "x0": [75, 200],

    # Bounds
    "lb": [50, 150],
    "ub": [100, 250],

    # Function names INSIDE the MATLAB file
    "objective_name": "obj_function",
    "constraint_name": "constraints",

    # Folder where MATLAB file lives
    "matlab_folder": os.path.join(os.path.dirname(__file__), "objective_functions"),

    # Optimization settings
    "algorithm": "sqp",
    "display": "iter",
    
    # SIMULATION
    
    
    # Input JSON File (PYSAM default input)
    "json_file": "simulation/Kanniyan_Project.json",
    
    "model":"PhysicalTroughNone",

    # overrides
    "overrides": {
        "T_startup": 75,
        "T_shutdown": 250,
        # "I_bn_des": 800,

        # # Tracking error values (example 1D array)
        # "TrackingError": [
        #     0.70,
        #     0.988,
        #     0.988,
        #     0.988
        # ],

        # Example of 2D array override (commented out)
        # "D_2": [
        #     [0.06, 0.05, 0.05, 0.05],
        #     [0.076, 0.076, 0.076, 0.076],
        #     [0.076, 0.076, 0.076, 0.076],
        #     [0.076, 0.076, 0.076, 0.076]
        # ],
    },
}
