import os
import json
import numpy as np
import PySAM.TroughPhysical as TP

from config import CONFIG



def run_simulation(overrides):
    
    # Convert MATLAB py arguments to Python
    overrides = dict(overrides)
    
    tp = TP.default(CONFIG["model"])
    
    # Load JSON
    with open(CONFIG["json_file"], "r") as f:
        data = json.load(f)
    for k, v in data.items():
        if k != "number_inputs":
            try:
                tp.value(k, v)
            except Exception:
                pass
    
    # Apply overrides (changed parameters)
    # overrides = CONFIG["overrides"]
    if overrides:
        for k, v in overrides.items():
            # print(f'Value before : {tp.value(k)}')
            tp.value(k, v)
            # print(f'Value after : {tp.value(k)}')
    # print('Simulation started...')
    tp.execute()
    # print('Simulation finished!')
    
    
    return float(tp.Outputs.annual_energy)   # dict of all outputs


