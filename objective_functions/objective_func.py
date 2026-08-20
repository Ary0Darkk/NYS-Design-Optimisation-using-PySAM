import pandas as pd
from functools import lru_cache
from pathlib import Path
from demand_data import get_dynamic_price
from config import CONFIG

timestamp = CONFIG["session_time"]


@lru_cache(maxsize=1)
def get_cached_dynamic_price():
    file_path = Path("electricity_data/dynamic_price_data.csv")

    if file_path.exists():
        df = pd.read_csv(file_path)
        return df["dynamic_price"].values

    else:
        return get_dynamic_price()["dynamic_price"].values


# calculate DAILY objective function
def objective_function(
    hourly_energy: list[float],
    field_htf_pump_power: list[float],
    pc_htf_pump_power: list[float],
    field_collector_tracking_power: list[float],
    pc_startup_thermal_power: list[float],
    field_piping_thermal_loss: list[float],
    receiver_thermal_loss: list[float],
    f_overrides: dict,
    day_index: int,
) -> float:
    """
    Calculates DAILY objective function using exact hourly aggregation.
    """

    # -----------------------------
    # build dataframe
    # -----------------------------
    data = {
        "hourly_energy": hourly_energy,
        "field_htf_pump_power": field_htf_pump_power,
        "pc_htf_pump_power": pc_htf_pump_power,
        "field_collector_tracking_power": field_collector_tracking_power,
        "pc_startup_thermal_power": pc_startup_thermal_power,
        "field_piping_thermal_loss": field_piping_thermal_loss,
        "receiver_thermal_loss": receiver_thermal_loss,
        "dynamic_price": get_cached_dynamic_price(),
    }

    df = pd.DataFrame(data)

    # -----------------------------
    # map day -> 24 hour slice
    # -----------------------------
    start = day_index * 24
    end = start + 24

    # -----------------------------
    # EXACT aggregation
    # multiply first -> then sum
    # -----------------------------

    hourly_energy_cost_term = sum(
        df["hourly_energy"][i] * df["dynamic_price"][i] * 1_000
        for i in range(start, end)
    )

    field_htf_pump_power_cost_term = sum(
        df["field_htf_pump_power"][i] * df["dynamic_price"][i] * 1_000
        for i in range(start, end)
    )

    pc_htf_pump_power_cost_term = sum(
        df["pc_htf_pump_power"][i] * df["dynamic_price"][i] * 1_000
        for i in range(start, end)
    )

    field_collector_tracking_power_cost_term = sum(
        df["field_collector_tracking_power"][i] * df["dynamic_price"][i] * 1_000
        for i in range(start, end)
    )

    pc_startup_thermal_power_cost_term = sum(
        df["pc_startup_thermal_power"][i] * df["dynamic_price"][i] * 1_000 * 0.4
        for i in range(start, end)
    )

    field_piping_thermal_loss_cost_term = sum(
        df["field_piping_thermal_loss"][i] * df["dynamic_price"][i] * 1_000 * 0.4
        for i in range(start, end)
    )

    receiver_thermal_loss_cost_term = sum(
        df["receiver_thermal_loss"][i] * df["dynamic_price"][i] * 1_000 * 0.4
        for i in range(start, end)
    )

    # -----------------------------
    # objective
    # -----------------------------
    obj = (
        hourly_energy_cost_term
        - field_htf_pump_power_cost_term
        - pc_htf_pump_power_cost_term
        - field_collector_tracking_power_cost_term
        - pc_startup_thermal_power_cost_term
        - field_piping_thermal_loss_cost_term
        - receiver_thermal_loss_cost_term
    )

    # -----------------------------
    # save override variables
    # -----------------------------
    var_data = pd.DataFrame([f_overrides])

    # -----------------------------
    # save monetary terms
    # -----------------------------
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

    terms_data["day"] = day_index

    terms_logbook = pd.DataFrame([terms_data])

    terms_logbook = pd.concat(
        [var_data.reset_index(drop=True), terms_logbook],
        axis=1,
    )

    terms_logbook = terms_logbook.set_index("day")

    # -----------------------------
    # save csv
    # -----------------------------
    terms_file_name = Path(f"results/terms/terms_data_{timestamp}.csv")

    terms_file_name.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = terms_file_name.exists()

    terms_logbook.to_csv(
        terms_file_name,
        mode="a",
        header=not file_exists,
    )

    return obj
