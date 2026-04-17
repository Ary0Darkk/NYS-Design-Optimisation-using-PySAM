import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
from pathlib import Path

from simulation.simulation import run_simulation
from demand_data import get_dynamic_price


class SolarConfig:
    USD_TO_INR = 73
    MWT_TO_MWE = 0.4
    LAND_EXPECTANCY_YEARS = 30
    KW_CONVERSION = 1000


class SolarOptimization:
    def __init__(self):
        self.config = SolarConfig()
        self.full_price_df = get_dynamic_price()
        self.dynamic_price_series = self.full_price_df["dynamic_price"]

        self.symbol_map = {
            "specified_total_aperture": "ΣA",
            "Row_Distance": "↔",
            "ColperSCA": "⊞",
            "W_aperture": "w",
            "L_SCA": "ℓ",
            "m_dot": "ṁ",
            "T_startup": "T↑",
            "T_shutdown": "T↓",
            "specified_solar_multiple": "SM",
        }

    def _prepare_dataframe(self, sim_result: dict) -> pd.DataFrame:
        data = {
            "hourly_energy": sim_result["hourly_energy"],
            "field_htf_pump_power": sim_result["field_htf_pump_power"],
            "pc_htf_pump_power": sim_result["pc_htf_pump_power"],
            "field_collector_tracking_power": sim_result[
                "field_collector_tracking_power"
            ],
            "pc_startup_thermal_power": sim_result["pc_startup_thermal_power"],
            "field_piping_thermal_loss": sim_result["field_piping_thermal_loss"],
            "receiver_thermal_loss": sim_result["receiver_thermal_loss"],
        }
        return pd.DataFrame(data).apply(lambda x: x.squeeze())

    def calculate_objective(self, sim_result: dict) -> dict:
        df = self._prepare_dataframe(sim_result)
        price_values = self.dynamic_price_series.values.flatten()

        hourly_energy = np.nan_to_num(df["hourly_energy"].values)

        elec_power = np.nan_to_num(
            df["field_htf_pump_power"].values
            + df["pc_htf_pump_power"].values
            + df["field_collector_tracking_power"].values
        )

        thermal_loss = np.nan_to_num(
            df["pc_startup_thermal_power"].values
            + df["field_piping_thermal_loss"].values
            + df["receiver_thermal_loss"].values
        )

        gross_energy = np.sum(hourly_energy * self.config.KW_CONVERSION * price_values)

        losses = (
            (elec_power * self.config.KW_CONVERSION)
            + (thermal_loss * self.config.MWT_TO_MWE * self.config.KW_CONVERSION)
        ) * price_values

        losses_total = np.sum(losses)

        total_installed_cost = float(sim_result["total_installed_cost"])
        penality = (
            total_installed_cost
            * self.config.USD_TO_INR
            / self.config.LAND_EXPECTANCY_YEARS
        )

        net_objective = gross_energy - losses_total - penality

        return {
            "Net Objective": net_objective,
            "Gross Value": gross_energy,
            "Losses": losses_total,
            "Penality": penality,
        }

    def run(self, sweep_config: dict):
        keys = list(sweep_config.keys())
        ranges = [np.arange(s, e + step, step) for s, e, step in sweep_config.values()]
        combinations = list(itertools.product(*ranges))

        sweep_data = []

        # Baseline (optional — won't be plotted in x-axis)
        base_res, _ = run_simulation(overrides={})
        base_metrics = self.calculate_objective(base_res)
        base_metrics["baseline"] = True
        sweep_data.append(base_metrics)

        # Sweep
        for combo in combinations:
            current_overrides = dict(zip(keys, combo))
            sim_res, _ = run_simulation(overrides=current_overrides)
            metrics = self.calculate_objective(sim_res)

            # ✅ STORE VARIABLE VALUES (CRUCIAL FIX)
            for k, v in current_overrides.items():
                metrics[k] = v

            metrics["baseline"] = False
            sweep_data.append(metrics)

        df = pd.DataFrame(sweep_data)
        self._plot_large_multi_graphs(df, sweep_config)

    def _plot_large_multi_graphs(self, df_sweep, sweep_config):
        metrics_to_plot = [
            ("Net Objective", "#1f77b4"),
            ("Gross Value", "#2ca02c"),
            ("Losses", "#ff7f0e"),
            ("Penality", "#d62728"),
        ]

        output_dir = Path("sensitivity_analysis_plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        var_name = list(sweep_config.keys())[0]
        symbol = self.symbol_map.get(var_name, var_name)

        df_plot = df_sweep[~df_sweep["baseline"]].copy()
        df_plot = df_plot.sort_values(by=var_name)

        x_values = df_plot[var_name].values

        for label, color in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)  # smaller, paper-friendly

            ax.plot(
                x_values,
                df_plot[label],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=color,
            )

            # SPARSE ANNOTATION (only every 3rd point)
            for i, (x, y) in enumerate(zip(x_values, df_plot[label])):
                # Alternate vertical offsets to avoid overlap
                offset = 6 if i % 2 == 0 else -10

                ax.annotate(
                    f"{y:.1e}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, offset),
                    ha="center",
                    va="bottom" if offset > 0 else "top",
                    fontsize=6,
                )

            # Labels (smaller font)
            ax.set_xlabel(f"{symbol} ({var_name})", fontsize=9)
            ax.set_ylabel("COST (INR)", fontsize=9)

            ax.set_title(
                f"{label} vs {var_name}",
                fontsize=10,
                fontweight="bold",
            )

            # Clean ticks
            ax.tick_params(axis="both", labelsize=8)

            # Grid (subtle)
            ax.grid(True, linestyle="--", alpha=0.4)

            # Remove clutter
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Tight layout for paper
            plt.tight_layout(pad=1)

            file_name = f"{var_name}_{label.lower().replace(' ', '_')}.png"
            plt.savefig(output_dir / file_name)
            plt.show()


if __name__ == "__main__":
    optimizer = SolarOptimization()

    # SINGLE VARIABLE SWEEP (STRICT)
    config = {"m_dot": (2, 12, 1)}

    optimizer.run(config)
