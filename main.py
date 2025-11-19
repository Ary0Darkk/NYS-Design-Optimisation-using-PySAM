import os
import json
import numpy as np
import PySAM.TroughPhysical as TP



def run_trough(json_file='Kanniyan_Project.json', overrides=None):
    
    tp = TP.default("PhysicalTroughNone")
    
    # Load JSON
    with open(json_file, "r") as f:
        data = json.load(f)
    for k, v in data.items():
        if k != "number_inputs":
            try:
                tp.value(k, v)
            except Exception:
                pass
    
    # Apply overrides (changed parameters)
    if overrides:
        for k, v in overrides.items():
            print(f'Value before : {tp.value(k)}')
            tp.value(k, v)
            print(f'Value after : {tp.value(k)}')
    print('Simulation started...')
    tp.execute()
    print('Simulation finished!')
    return tp.Outputs.export()   # dict of all outputs


in_overrides = {
    'T_startup':75,
    'T_shutdown': 250,
    'I_bn_des':800,
    "TrackingError": [
    0.7, 0.98799999999999999, 0.98799999999999999, 0.98799999999999999],
    # "D_2": [
    # [
    #   0.060000000000000003, 0.050000000000000003, 0.050000000000000003,
    #   0.050000000000000003
    # ],
    # [
    #   0.075999999999999998, 0.075999999999999998, 0.075999999999999998,
    #   0.075999999999999998
    # ],
    # [
    #   0.075999999999999998, 0.075999999999999998, 0.075999999999999998,
    #   0.075999999999999998
    # ],
    # [
    #   0.075999999999999998, 0.075999999999999998, 0.075999999999999998,
    #   0.075999999999999998
    # ]
    # ]

}
output = run_trough(overrides=in_overrides)

print(f'Annual energy :{output["annual_energy"]}')