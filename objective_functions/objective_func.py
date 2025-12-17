import pandas as pd
from demand_data import get_dynamic_price


# TODO : write correct code for this func
def objective_function(
    hourly_energy: list[float],
    field_htf_pump_power: list[float],
    pc_htf_pump_power: list[float],
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
    hourly_energy_df = pd.Series(hourly_energy, name="hourly_energy")
    print(hourly_energy_df.head())
    field_htf_pump_power_df = pd.Series(
        field_htf_pump_power, name="field_htf_pump_power"
    )
    print(field_htf_pump_power_df.head())
    pc_htf_pump_power_df = pd.Series(pc_htf_pump_power, name="pc_htf_pump_power")
    print(pc_htf_pump_power_df.head())

    # access final dataset
    final_dataset = get_dynamic_price()
    dynamic_price_df = final_dataset["dynamic price"]
    print(dynamic_price_df.head())

    obj = (
        hourly_energy_df * dynamic_price_df
        - field_htf_pump_power_df * dynamic_price_df
        - pc_htf_pump_power_df * dynamic_price_df
    )

    print(obj.head())

    return obj
