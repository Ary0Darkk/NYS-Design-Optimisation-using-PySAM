import pandas as pd
from pathlib import Path
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
        "dynamic_price": get_dynamic_price()["dynamic_price"].values,  # Rs./KWh
    }

    # Create the DataFrame all at once
    df = pd.DataFrame(data)

    # Shift the index to start at 1
    df.index = df.index + 1

    # If you need to access them individually later:
    # hourly_energy_df = df["hourly_energy"]

    hourly_energy_term = (
        df["hourly_energy"][hour_index] * df["dynamic_price"][hour_index] * 1_000
    )
    field_htf_pump_power_term = (
        df["field_htf_pump_power"][hour_index] * df["dynamic_price"][hour_index] * 1_000
    )
    pc_htf_pump_power_term = (
        df["pc_htf_pump_power"][hour_index] * df["dynamic_price"][hour_index] * 1_000
    )
    field_collector_tracking_power_term = (
        df["field_collector_tracking_power"][hour_index]
        * df["dynamic_price"][hour_index]
        * 1_000
    )
    pc_startup_thermal_power_term = (
        df["pc_startup_thermal_power"][hour_index]
        * df["dynamic_price"][hour_index]
        * 1_000
        * 0.4
    )
    field_piping_thermal_loss_term = (
        df["field_piping_thermal_loss"][hour_index]
        * df["dynamic_price"][hour_index]
        * 1_000
        * 0.4
    )
    receiver_thermal_loss_term = (
        df["receiver_thermal_loss"][hour_index]
        * df["dynamic_price"][hour_index]
        * 1_000
        * 0.4
    )
    obj = (
        hourly_energy_term  # gross term followed by other penality terms
        - field_htf_pump_power_term
        - pc_htf_pump_power_term
        - field_collector_tracking_power_term
        - pc_startup_thermal_power_term
        - field_piping_thermal_loss_term
        - receiver_thermal_loss_term
    )
    terms_data = {}
    terms_data["objective_fn_value"] = obj
    terms_data["hourly_energy_term"] = hourly_energy_term
    terms_data["field_htf_pump_power_term"] = field_htf_pump_power_term
    terms_data["pc_htf_pump_power_term"] = pc_htf_pump_power_term
    terms_data["field_collector_tracking_power_term"] = (
        field_collector_tracking_power_term
    )
    terms_data["pc_startup_thermal_power_term"] = pc_startup_thermal_power_term
    terms_data["field_piping_thermal_loss_term"] = field_piping_thermal_loss_term
    terms_data["receiver_thermal_loss_term"] = receiver_thermal_loss_term
    terms_data["hour"] = hour_index
    terms_logbook = pd.DataFrame([terms_data])

    terms_logbook = terms_logbook.set_index("hour")

    file_name = Path("results/terms_data.csv")
    file_name.parent.mkdir(exist_ok=True)

    file_exists = file_name.exists()
    terms_logbook.to_csv(file_name, mode="a", header=not file_exists)

    return obj
