import pandas as pd
from functools import cache
from pathlib import Path
from demand_data import get_dynamic_price


@cache
def get_cached_dynamic_price():
    file_path = Path("electricity_data/dynamic_price_data")
    if file_path.exists():
        df = pd.read_csv(file_path)
        return df["dynamic_price"].values
    else:
        return get_dynamic_price()["dynamic_price"].values


# calculate objective function value
def objective_function(
    hourly_energy: list[float],
    field_htf_pump_power: list[float],
    pc_htf_pump_power: list[float],
    field_collector_tracking_power: list[float],
    pc_startup_thermal_power: list[float],
    field_piping_thermal_loss: list[float],
    receiver_thermal_loss: list[float],
    hour_index: int,
) -> float:
    """
    Calculates the objective function for optimisation

    """

    # convert them into df
    data = {
        "hourly_energy": hourly_energy,  # MWe
        "field_htf_pump_power": field_htf_pump_power,  # MWe
        "pc_htf_pump_power": pc_htf_pump_power,  # MWe
        "field_collector_tracking_power": field_collector_tracking_power,  # MWe
        "pc_startup_thermal_power": pc_startup_thermal_power,  # MWt
        "field_piping_thermal_loss": field_piping_thermal_loss,  # MWt
        "receiver_thermal_loss": receiver_thermal_loss,  # MWt
        "dynamic_price": get_cached_dynamic_price(),  # Rs./KWh
    }

    # Create the DataFrame all at once
    df = pd.DataFrame(data)

    # Shift the index to start at 1
    df.index = df.index + 1

    # ---- init terms ----
    hourly_energy_value = df["hourly_energy"][hour_index]
    hourly_energy_cost_term = (
        hourly_energy_value * df["dynamic_price"][hour_index] * 1_000
    )
    field_htf_pump_power_value = df["field_htf_pump_power"][hour_index]
    field_htf_pump_power_cost_term = (
        field_htf_pump_power_value * df["dynamic_price"][hour_index] * 1_000
    )
    pc_htf_pump_power_value = df["pc_htf_pump_power"][hour_index]
    pc_htf_pump_power_cost_term = (
        pc_htf_pump_power_value * df["dynamic_price"][hour_index] * 1_000
    )
    field_collector_tracking_power_value = df["field_collector_tracking_power"][
        hour_index
    ]
    field_collector_tracking_power_cost_term = (
        field_collector_tracking_power_value * df["dynamic_price"][hour_index] * 1_000
    )
    pc_startup_thermal_power_value = df["pc_startup_thermal_power"][hour_index]
    pc_startup_thermal_power_cost_term = (
        pc_startup_thermal_power_value * df["dynamic_price"][hour_index] * 1_000 * 0.4
    )
    field_piping_thermal_loss_value = df["field_piping_thermal_loss"][hour_index]
    field_piping_thermal_loss_cost_term = (
        field_piping_thermal_loss_value * df["dynamic_price"][hour_index] * 1_000 * 0.4
    )
    receiver_thermal_loss_value = df["receiver_thermal_loss"][hour_index]
    receiver_thermal_loss_cost_term = (
        receiver_thermal_loss_value * df["dynamic_price"][hour_index] * 1_000 * 0.4
    )
    dynamic_price_value = df["dynamic_price"][hour_index]

    # ---- objective function ----
    obj = (
        hourly_energy_cost_term  # gross term followed by other penality terms
        - field_htf_pump_power_cost_term
        - pc_htf_pump_power_cost_term
        - field_collector_tracking_power_cost_term
        - pc_startup_thermal_power_cost_term
        - field_piping_thermal_loss_cost_term
        - receiver_thermal_loss_cost_term
    )

    # ---- save value of terms ----
    values_data = {}  # init dict
    values_data["dynamic_price_value"] = dynamic_price_value
    values_data["hourly_energy_value"] = hourly_energy_value
    values_data["field_htf_pump_power_value"] = field_htf_pump_power_value
    values_data["pc_htf_pump_power_value"] = pc_htf_pump_power_value
    values_data["field_collector_tracking_power_value"] = (
        field_collector_tracking_power_value
    )
    values_data["pc_startup_thermal_power_value"] = pc_startup_thermal_power_value
    values_data["field_piping_thermal_loss_value"] = field_piping_thermal_loss_value
    values_data["receiver_thermal_loss_value"] = receiver_thermal_loss_value
    values_data["hour"] = hour_index
    value_data_logbook = pd.DataFrame([values_data])

    value_data_logbook = value_data_logbook.set_index("hour")

    value_data_file_name = Path("results/value_data.csv")
    value_data_file_name.parent.mkdir(exist_ok=True)

    value_file_exists = value_data_file_name.exists()
    value_data_logbook.to_csv(
        value_data_file_name, mode="a", header=not value_file_exists
    )

    # ---- save complete term values in monetry unit ----
    terms_data = {}
    terms_data["objective_fn_value"] = obj
    terms_data["hourly_energy_term"] = hourly_energy_cost_term
    terms_data["field_htf_pump_power_term"] = field_htf_pump_power_cost_term
    terms_data["pc_htf_pump_power_term"] = pc_htf_pump_power_cost_term
    terms_data["field_collector_tracking_power_term"] = (
        field_collector_tracking_power_cost_term
    )
    terms_data["pc_startup_thermal_power_term"] = pc_startup_thermal_power_cost_term
    terms_data["field_piping_thermal_loss_term"] = field_piping_thermal_loss_cost_term
    terms_data["receiver_thermal_loss_term"] = receiver_thermal_loss_cost_term
    terms_data["hour"] = hour_index
    terms_logbook = pd.DataFrame([terms_data])

    terms_logbook = terms_logbook.set_index("hour")

    terms_file_name = Path("results/terms_data.csv")
    terms_file_name.parent.mkdir(exist_ok=True)

    file_exists = terms_file_name.exists()
    terms_logbook.to_csv(terms_file_name, mode="a", header=not file_exists)

    # return objective function value back
    return obj
