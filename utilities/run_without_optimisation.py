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
            # --- Design Parameters ---
            "specified_total_aperture": "ΣA",  # Total aperture area
            "Row_Distance": "↔",  # Row Distance
            "ColperSCA": "⊞",  # num of Modules per SCA
            "W_aperture": "w",  # Width of Aperture
            "L_SCA": "ℓ",  # length of Collector
            # --- Operational Parameters ---
            "m_dot": "ṁ",  # mass flow rate
            "T_startup": "T↑",  # Startup Temperature
            "T_shutdown": "T↓",  # Shutdown Temperature
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

        # Use nan_to_num to treat hourly errors as 0 penalty/energy
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

        # --- Baseline Calculation (Included in Plot) ---
        base_res, _ = run_simulation(overrides={})
        base_metrics = self.calculate_objective(base_res)
        base_metrics["bracket_label"] = "Baseline"
        sweep_data.append(base_metrics)

        # --- Sweep Calculation ---
        for combo in combinations:
            current_overrides = dict(zip(keys, combo))
            sim_res, _ = run_simulation(overrides=current_overrides)
            metrics = self.calculate_objective(sim_res)

            label_parts = [
                f"{self.symbol_map.get(k, k)}: {v}"
                for k, v in current_overrides.items()
            ]
            metrics["bracket_label"] = f"({', '.join(label_parts)})"
            sweep_data.append(metrics)

        self._plot_large_multi_graphs(pd.DataFrame(sweep_data), sweep_config)

    def _plot_large_multi_graphs(self, df_sweep, sweep_config):
        metrics_to_plot = [
            ("Net Objective", "#1f77b4"),
            ("Gross Value", "#2ca02c"),
            ("Losses", "#ff7f0e"),
            ("Penality", "#d62728"),
        ]
        # Define and create the output directory using pathlib
        output_dir = Path("sensitivity_analysis_plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        config_names = "_".join(sweep_config.keys())

        for label, color in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
            x_indices = np.arange(len(df_sweep))

            ax.plot(
                x_indices,
                df_sweep[label],
                color=color,
                marker="o",
                linewidth=2,
                markersize=8,
                alpha=0.8,
                label=label,
            )

            # Tight Annotations (Raw Y-values)
            for i, row in df_sweep.iterrows():
                y_val = row[label]

                # Combine parameter label and raw numerical result
                annotation_text = f"{row['bracket_label']}\n{y_val}"

                # Alternate offset to prevent overlap; slightly larger for 2-line text
                y_offset = 18 if i % 2 == 0 else -28

                ax.annotate(
                    annotation_text,
                    xy=(i, y_val),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    fontsize=7,
                    ha="center",
                    va="bottom" if y_offset > 0 else "top",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white", ec=color, lw=1, alpha=0.8
                    ),
                )

            ax.set_xticks(x_indices)
            # Label S1 as Baseline
            xtick_labels = [
                f"S{i + 1}\n(Base)" if i == 0 else f"S{i + 1}" for i in x_indices
            ]
            ax.set_xticklabels(xtick_labels, fontsize=9)

            ax.margins(x=0.05, y=0.18)
            ax.set_title(f"Analysis: {label}", fontsize=14, fontweight="bold", pad=15)
            ax.set_ylabel("COST (in INR)", fontsize=10)
            ax.grid(True, axis="y", linestyle=":", alpha=0.5)

            # Build legend with symbols
            handles, labels = ax.get_legend_handles_labels()
            for k in sweep_config.keys():
                symbol = self.symbol_map.get(k, k)
                handles.append(
                    plt.Line2D([0], [0], color="none", label=f"{symbol}: {k}")
                )

            # Place legend at the top, spread across 4 columns to save vertical space
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                ncol=3,
                fontsize=8,
                frameon=True,
                edgecolor="gray",
                title="Scenario Key",
                title_fontsize="9",
            )

            plt.tight_layout()
            clean_label = label.lower().replace(" ", "_")
            file_name = f"{config_names}_analysis_{clean_label}.png"
            save_path = output_dir / file_name
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved: {save_path}")
            plt.show()


if __name__ == "__main__":
    optimizer = SolarOptimization()
    config = {"Row_Distance": (5, 30, 5)}
    optimizer.run(config)
