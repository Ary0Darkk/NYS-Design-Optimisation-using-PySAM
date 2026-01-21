import pandas as pd
from demand_data import get_dynamic_price


# calculate objective function value
def objective_function(
    hourly_energy: list[float],
    field_htf_pump_power: list[float],
    pc_htf_pump_power: list[float],
    field_collector_tracking_power: list[float],
    pc_startup_thermal_power: list[float],
    field_piping_thermal_loss: list[float],
    receiver_thermal_loss: list[float],
    parasitic_power_generation_dependent_load: list[float],
    field_collector_row_shadowing_loss: list[float],
    parasitic_power_fixed_load: list[float],
    parasitic_power_condenser_operation: list[float],
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
        "field_htf_pump_power": field_htf_pump_power,  # MWe
        "pc_htf_pump_power": pc_htf_pump_power,  # MWe
        "field_collector_tracking_power": field_collector_tracking_power,  # MWe
        "pc_startup_thermal_power": pc_startup_thermal_power,  # MWt
        "field_piping_thermal_loss": field_piping_thermal_loss,  # MWt
        "receiver_thermal_loss": receiver_thermal_loss,  # MWt
        "parasitic_power_generation_dependent_load": parasitic_power_generation_dependent_load,  # MWe
        "field_collector_row_shadowing_loss": field_collector_row_shadowing_loss,  # fraction
        "parasitic_power_fixed_load": parasitic_power_fixed_load,  # MWe
        "parasitic_power_condenser_operation": parasitic_power_condenser_operation,  # MWe
        "dynamic_price": get_dynamic_price()["dynamic_price"].values,
    }

    # Create the DataFrame all at once
    df = pd.DataFrame(data)

    # Shift the index to start at 1
    df.index = df.index + 1

    # If you need to access them individually later:
    # hourly_energy_df = df["hourly_energy"]

    obj = (
        df["hourly_energy"][hour_index] * df["dynamic_price"][hour_index]
        - df["field_htf_pump_power"][hour_index] * df["dynamic_price"][hour_index]
        - df["pc_htf_pump_power"][hour_index] * df["dynamic_price"][hour_index]
        - df["field_collector_tracking_power"][hour_index]
        * df["dynamic_price"][hour_index]
        - df["pc_startup_thermal_power"][hour_index] * df["dynamic_price"][hour_index]
        - df["field_piping_thermal_loss"][hour_index] * df["dynamic_price"][hour_index]
        - df["receiver_thermal_loss"][hour_index] * df["dynamic_price"][hour_index]
        - df["parasitic_power_generation_dependent_load"][hour_index]
        * df["dynamic_price"][hour_index]
        - df["field_collector_row_shadowing_loss"][hour_index]
        * df["dynamic_price"][hour_index]
        - df["parasitic_power_fixed_load"][hour_index] * df["dynamic_price"][hour_index]
        - df["parasitic_power_condenser_operation"][hour_index]
        * df["dynamic_price"][hour_index]
    ) * 1_000

    return obj
