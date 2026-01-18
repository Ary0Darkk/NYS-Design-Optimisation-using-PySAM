import json
import PySAM.TroughPhysical as TP
import hashlib

from config import CONFIG
from utilities.list_nesting import replace_1st_order

from prefect import task
from prefect.logging import get_run_logger
from datetime import timedelta


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
                else round(float(v), 6),
            )
            for k, v in overrides.items()
        )
    )


def simulation_cache_key(context, parameters):
    canon = canonicalize_overrides(parameters["overrides"])
    digest = hashlib.sha256(repr(canon).encode()).hexdigest()
    return f"sim_{digest}"


@task(
    cache_key_fn=simulation_cache_key,
    persist_result=True,
    cache_expiration=timedelta(days=30),
    result_storage=CONFIG["storage_block"],
)
def run_simulation(overrides):
    logger = get_run_logger()

    overrides = dict(overrides)

    tp = TP.default(CONFIG["model"])
    logger.info(f"{CONFIG['model']} model loaded!")

    # Load JSON
    with open(CONFIG["json_file"], "r") as f:
        data = json.load(f)
    logger.info(f"{CONFIG['json_file']} file loaded!")

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

    logger.info("Variables assigned from json file to model!")

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
    logger.info("Custom overrides of variables performed!")
    logger.info("Simulation started...")
    tp.execute()
    logger.info("Simulation finished!")

    sim_result = {
        "hourly_energy": tp.Outputs.P_out_net,  # FIXME: I think here I should take PC electrical power output : P_cycle as P_out_net is net, not gross
        "pc_htf_pump_power": tp.Outputs.cycle_htf_pump_power,
        "field_htf_pump_power": tp.Outputs.W_dot_field_pump,
        "field_collector_tracking_power": tp.Outputs.W_dot_sca_track,
        "pc_startup_thermal_power": tp.Outputs.q_dot_pc_startup,
        "field_piping_thermal_loss": tp.Outputs.q_dot_piping_loss,
        "receiver_thermal_loss": tp.Outputs.q_dot_rec_thermal_loss,
        "parasitic_power_generation_dependent_load": tp.Outputs.P_plant_balance_tot,
        "field_collector_row_shadowing_loss": tp.Outputs.RowShadow_ave,
        "parasitic_power_fixed_load": tp.Outputs.P_fixed,
        "parasitic_power_condenser_operation": tp.Outputs.P_cooling_tower_tot,
        # NOTE: I am not taking Field collector optical end loss:EndLoss_ave for now!
        "annual_energy": tp.Outputs.annual_energy,
    }

    logger.info("Outputs written to sim_result dict!")
    return sim_result  # dict of outputs
