import json
import os
import PySAM.TroughPhysical as TP
import hashlib
import tabulate as tb
import time

from config import CONFIG
from utilities.list_nesting import replace_1st_order
from utilities.setup_custom_logger import setup_custom_logger

import logging
from joblib import Memory, hash as joblib_hash

# Set up logging if not already done
if not logging.getLogger("NYS_Optimisation").hasHandlers():
    logger = setup_custom_logger()
else:
    logger = logging.getLogger("NYS_Optimisation")

# Define codes
NEW = "\033[0;36m"  # Cyan
CACHED = "\033[0;32m"  # Green
RESET = "\033[0m"  # No Color


def canonicalize_overrides(overrides):
    """this canonicalize the overrides

    Args:
        overrides (dict): variables that need to overriden

    Returns:
        tuple: contains int and float based on type defined in override
    """
    return tuple(
        sorted(
            (
                k,
                int(v)
                if isinstance(v, bool) or float(v).is_integer()
                else round(float(v), 2),
            )
            for k, v in overrides.items()
        )
    )


def simulation_cache_key(parameters):
    canon = parameters["overrides"]
    digest = hashlib.sha256(repr(canon).encode()).hexdigest()
    return f"sim_{digest}"


# setup the cache directory (it will be created automatically)
# Setting mmap_mode='r' makes it very fast for large arrays
cachedir = os.path.join(os.getcwd(), "sim_cache")
memory = Memory(cachedir, verbose=0, mmap_mode="r")


def run_simulation(overrides: dict):
    """
    The main entry point. It checks the cache status,
    executes the core, and logs dynamically.
    """
    # check if it's already on disk
    # use joblib_hash and the internal joblib folder structure
    # arg_hash = joblib_hash(dict(overrides))

    # Note: 'simulation' is the name of this file (simulation.py)
    # If this file is named engine.py, change 'simulation' to 'engine'
    # path_parts = [
    #     cachedir, 
    #     "joblib", 
    #     "simulation", 
    #     "simulation", 
    #     "_run_simulation_core", 
    #     arg_hash, 
    #     "output.pkl"
    # ]
    # cache_path = os.path.join(*path_parts)
    # cache_path = os.path.join(
    #     cachedir,
    #     "joblib",
    #     "simulation",
    #     "simulation",
    #     "_run_simulation_core",
    #     arg_hash,
    #     "output.pkl",
    # )

    # is_cached = os.path.exists(cache_path)

    start = time.time()
    # run the actual simulation (joblib handles the loading)
    result = _run_simulation_core(overrides)
    duration = time.time() - start

    is_cached = duration < 0.1

    # dynamic Logging & Table
    status_tag = f"{CACHED}[CACHED]{RESET}" if is_cached else f"{NEW}[NEW RUN]{RESET}"

    table = tb.tabulate([overrides.values()], headers=overrides.keys(), tablefmt="psql")

    # log message changes dynamically
    logger.info(f"{status_tag} Simulation processed with parameters:\n{table}")

    return result


@memory.cache
def _run_simulation_core(overrides: dict):
    overrides = dict(overrides)

    # handle mass flow rate values
    for key, value in overrides.items():
        if key == "m_dot":
            overrides["m_dot_htfmin"] = value
            overrides["m_dot_htfmax"] = value
            del overrides["m_dot"]
            break

    tp = TP.default(CONFIG["model"])
    # logger.info(f"{CONFIG['model']} model loaded!")

    # Load JSON
    with open(CONFIG["json_file"], "r") as f:
        data = json.load(f)
    # logger.info(f"{CONFIG['json_file']} file loaded!")

    # assign(dict) -> None : takes dict and copies the values into the PySAM model
    # export() -> dict : pulls every single parameter currently set in that
    # PySAM module and puts them into a standard Python dictionary
    # replace(dict) -> None : If your dictionary only has 5 variables,
    # PySAM will "unassign" or clear all other variables in that module that are not in your dictionary.
    # NOTE : can't use assign here, because we are assigning
    # variables of different sub-modules
    for k, v in data.items():
        if k != "number_inputs":
            try:
                tp.value(k, v)
            except Exception:
                pass

    # logger.info("Variables assigned from json file to model!")

    # Apply overrides (changed parameters)
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
    # logger.info("Custom overrides of variables performed!")
    # logger.info("Simulation started...")
    tp.execute()
    # logger.info("Simulation finished!")

    sim_result = {
        "hourly_energy": tp.Outputs.P_cycle,  # FIXME: I think here I should take PC electrical power output : P_cycle as P_out_net is net, not gross
        "pc_htf_pump_power": tp.Outputs.cycle_htf_pump_power,
        "field_htf_pump_power": tp.Outputs.W_dot_field_pump,
        "field_collector_tracking_power": tp.Outputs.W_dot_sca_track,
        "pc_startup_thermal_power": tp.Outputs.q_dot_pc_startup,
        "field_piping_thermal_loss": tp.Outputs.q_dot_piping_loss,
        "receiver_thermal_loss": tp.Outputs.q_dot_rec_thermal_loss,
        # "field_collector_row_shadowing_loss": tp.Outputs.RowShadow_ave,
        # "parasitic_power_generation_dependent_load": tp.Outputs.P_plant_balance_tot,
        # "parasitic_power_fixed_load": tp.Outputs.P_fixed,
        # "parasitic_power_condenser_operation": tp.Outputs.P_cooling_tower_tot,
        # NOTE: I am not taking Field collector optical end loss:EndLoss_ave for now!
        "annual_energy": tp.Outputs.annual_energy,
    }

    # logger.info("Outputs written to sim_result dict!")
    return sim_result  # dict of outputs
