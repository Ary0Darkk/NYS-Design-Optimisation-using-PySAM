import json
import PySAM.TroughPhysical as TP

from config import CONFIG
from utilities.list_nesting import replace_1st_order

from prefect import task
from prefect.tasks import task_input_hash
from prefect.logging import get_run_logger
from datetime import timedelta


@task(
    cache_key_fn=task_input_hash,
    persist_result=True,
    cache_expiration=timedelta(days=1),
    result_storage=CONFIG["storage_block"],
)
def run_simulation(overrides):
    logger = get_run_logger()
    # Convert MATLAB py arguments to Python
    overrides = dict(overrides)

    tp = TP.default(CONFIG["model"])
    logger.info(f"{CONFIG['model']} model loaded!")

    # Load JSON
    with open(CONFIG["json_file"], "r") as f:
        data = json.load(f)
    logger.info(f"{CONFIG['json_file']} file loaded in json format!")
    for k, v in data.items():
        if k != "number_inputs":
            try:
                tp.value(k, v)
            except Exception:
                pass
    logger.info("Variables assigned from json file to model!")
    # Apply overrides (changed parameters)
    # overrides = CONFIG["overrides"]
    if overrides:
        for k, v in overrides.items():
            # print(f'Value before : {tp.value(k)}')
            # Fetch current value once (Best practice for performance)
            current_val = tp.value(k)

            # Case: Scalar (int, float, or boolean)
            if isinstance(current_val, (int, float, bool)):
                tp.value(k, v)

            #  Case: 1st Order Sequence (List or Tuple)
            elif isinstance(current_val, (list, tuple)):
                if len(current_val) > 0:
                    # Use your 1st order function to swap the first element
                    v_new = replace_1st_order(data=current_val, new_val=v)
                    tp.value(k, v_new)
                else:
                    # If the list is empty, we initialize it with the new value
                    tp.value(k, [v] if isinstance(current_val, list) else (v,))

            # 4. Optional: Log or skip if it's an unexpected type (like a string)
            else:
                print(f"Skipping key {k}: Unknown type {type(current_val)}")
            # print(f'Value after : {tp.value(k)}')
    logger.info("Custom overrides of variables performed!")
    logger.info("Simulation started...")
    tp.execute()
    logger.info("Simulation finished!")

    sim_result = {
        "hourly_energy": tp.Outputs.P_out_net,
        "pc_htf_pump_power": tp.Outputs.cycle_htf_pump_power,
        "field_htf_pump_power": tp.Outputs.W_dot_field_pump,
    }

    logger.info("Outputs written to sim_result dict!")
    return sim_result  # dict of outputs
