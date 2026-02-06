import json
import PySAM.TroughPhysical as TP
import tabulate as tb
import time

from config import CONFIG
from utilities.list_nesting import replace_1st_order
from utilities.setup_custom_logger import setup_custom_logger

import logging

# Set up logging if not already done
if not logging.getLogger("NYS_Optimisation").hasHandlers():
    logger = setup_custom_logger()
else:
    logger = logging.getLogger("NYS_Optimisation")

# Define codes
NEW = "\033[0;36m"  # Cyan
CACHED = "\033[0;32m"  # Green
RESET = "\033[0m"  # No Color
LIGHT_GRAY = "\033[2m"
CRITICAL = "\033[0;31m"  # red

# Load JSON
with open(CONFIG["json_file"], "r") as f:
    data = json.load(f)


def run_simulation(overrides: dict):
    """
    The main entry point. It executes the core,
    and logs dynamically
    """
    # overrides = dict(sorted(overrides.items())) # sorted to ensure same order

    duration = 0
    result = None
    try:
        start = time.time()
        # run the actual simulation (joblib handles the loading)
        result = _run_simulation_core(overrides)
        penalty_flag = False  # flag for penality; pysam model not executed
        duration = time.time() - start
        # Convert seconds to minutes and seconds
        mins, secs = divmod(duration, 60)
        # log message
        table = tb.tabulate(
            [overrides.values()], headers=overrides.keys(), tablefmt="psql"
        )
        logger.info(
            f"{NEW}[NEW RUN]{RESET} [{int(mins)}m {secs:05.2f}s] Ran sim with params :\n{table}"
        )
    except Exception as e:
        logger.critical(f"Sim exited with params : {overrides}")
        logger.critical(f"Model exited with error: {e}")
        penalty_flag = True
        logger.info(
            f"{CRITICAL}[PENALISED]{RESET}Penalised with {CONFIG['penalty']:.0e} penality"
        )

    return result, penalty_flag


def _run_simulation_core(overrides: dict):
    overrides = dict(overrides)

    # handle mass flow rate values
    for key, value in overrides.items():
        if key == "m_dot":
            overrides["m_dot_htfmin"] = value
            overrides["m_dot_htfmax"] = value
            del overrides["m_dot"]
            break

    # loads model with default values
    model = TP.default(CONFIG["model"])

    # initial assignment of all variables from JSON
    for k, v in data.items():
        if k != "number_inputs":
            try:
                model.value(k, v)
            except Exception:
                pass

    # tp = TP.default(CONFIG["model"])
    # logger.info(f"{CONFIG['model']} model loaded!")

    # logger.info(f"{CONFIG['json_file']} file loaded!")

    # assign(dict) -> None : takes dict and copies the values into the PySAM model
    # export() -> dict : pulls every single parameter currently set in that
    # PySAM module and puts them into a standard Python dictionary
    # replace(dict) -> None : If your dictionary only has 5 variables,
    # PySAM will "unassign" or clear all other variables in that module that are not in your dictionary.
    # NOTE : can't use assign here, because we are assigning
    # variables of different sub-modules

    # logger.info("Variables assigned from json file to model!")

    # Apply overrides (changed parameters)
    if overrides:
        for k, v in overrides.items():
            # print(f'Value before : {tp.value(k)}')
            # Fetch current value once (Best practice for performance)
            current_val = model.value(k)

            # Case: Scalar (int, float, or boolean)
            if isinstance(current_val, (int, float, bool)):
                model.value(k, v)

            #  Case: 1st Order Sequence (List or Tuple)
            elif isinstance(current_val, (list, tuple)):
                if len(current_val) > 0:
                    # Use your 1st order function to swap the first element
                    v_new = replace_1st_order(data=current_val, new_val=v)
                    model.value(k, v_new)
                else:
                    # If the list is empty, we initialize it with the new value
                    model.value(k, [v] if isinstance(current_val, list) else (v,))

            # 4. Optional: Log or skip if it's an unexpected type (like a string)
            else:
                print(f"Skipping key {k}: Unknown type {type(current_val)}")
            # print(f'Value after : {tp.value(k)}')
    # logger.info("Custom overrides of variables performed!")
    # logger.info("Simulation started...")
    model.execute()
    # logger.info("Simulation finished!")

    sim_result = {
        "hourly_energy": model.Outputs.P_cycle,  # FIXME: I think here I should take PC electrical power output : P_cycle as P_out_net is net, not gross
        "pc_htf_pump_power": model.Outputs.cycle_htf_pump_power,
        "field_htf_pump_power": model.Outputs.W_dot_field_pump,
        "field_collector_tracking_power": model.Outputs.W_dot_sca_track,
        "pc_startup_thermal_power": model.Outputs.q_dot_pc_startup,
        "field_piping_thermal_loss": model.Outputs.q_dot_piping_loss,
        "receiver_thermal_loss": model.Outputs.q_dot_rec_thermal_loss,
        # "field_collector_row_shadowing_loss": tp.Outputs.RowShadow_ave,
        # "parasitic_power_generation_dependent_load": tp.Outputs.P_plant_balance_tot,
        # "parasitic_power_fixed_load": tp.Outputs.P_fixed,
        # "parasitic_power_condenser_operation": tp.Outputs.P_cooling_tower_tot,
        # NOTE: I am not taking Field collector optical end loss:EndLoss_ave for now!
        "annual_energy": model.Outputs.annual_energy,
    }

    # logger.info("Outputs written to sim_result dict!")
    return sim_result  # dict of outputs
