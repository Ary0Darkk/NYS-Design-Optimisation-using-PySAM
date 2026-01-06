import pandas as pd
from demand_data import get_dynamic_price
from prefect import task
from prefect.tasks import task_input_hash
from datetime import timedelta

from config import CONFIG


# TODO : write correct code for this func
# @task(
#     cache_key_fn=task_input_hash,
#     persist_result=True,
#     cache_expiration=timedelta(days=1),
#     result_storage=CONFIG["storage_block"],
# )
def objective_function(
    hourly_energy: list[float],
    field_htf_pump_power: list[float],
    pc_htf_pump_power: list[float],
    hour_index: int,
) -> float:
    """
    Calculates the objective function for optimisation

    :param hourly_energy: hourly energy
    :type hourly_energy: list[float]
    :param field_htf_pump_power: hourly field htf pumping power
    :type field_htf_pump_power: list[float]
    :param pc_htf_pump_power: hourly power cycle htf pumping power
    :type pc_htf_pump_power: list[float]
    """

    # convert them into df
    data = {
        "hourly_energy": hourly_energy,
        "field_htf_pump_power": field_htf_pump_power,
        "pc_htf_pump_power": pc_htf_pump_power,
        "dynamic_price": get_dynamic_price()["dynamic_price"].values,
    }

    # 2. Create the DataFrame all at once
    df = pd.DataFrame(data)

    # 3. Shift the index to start at 1
    df.index = df.index + 1

    # If you need to access them individually later:
    # hourly_energy_df = df["hourly_energy"]

    obj = (
        df["hourly_energy"][hour_index] * df["dynamic_price"][hour_index]
        - df["field_htf_pump_power"][hour_index] * df["dynamic_price"][hour_index]
        - df["pc_htf_pump_power"][hour_index] * df["dynamic_price"][hour_index]
    ) * 1_000

    return obj
