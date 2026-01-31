import torch
import torch.nn as nn
import numpy as np
from simulation import run_simulation
from objective_functions import objective_function
from config import CONFIG


class GaussianSearchPolicy(nn.Module):
    """
    πθ(x) = N(mu, diag(std^2))
    """

    def __init__(self, dim, init_std=0.3):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_std = nn.Parameter(torch.log(torch.ones(dim) * init_std))

    def sample(self, n):
        std = self.log_std.exp()
        dist = torch.distributions.Normal(self.mu, std)
        x = dist.rsample((n,))
        log_prob = dist.log_prob(x).sum(dim=1)
        return x, log_prob


def evaluate_candidate(x, optim_mode, hour_index, static_overrides, var_names):
    overrides_dyn = {k: float(v) for k, v in zip(var_names, x)}
    final_overrides = {**overrides_dyn, **static_overrides}

    sim_result, penalty_flag = run_simulation(final_overrides)

    if penalty_flag:
        return CONFIG["penalty"]

    if optim_mode == "design":
        return sim_result["annual_energy"]

    elif optim_mode == "operational":
        return objective_function(
            sim_result["hourly_energy"],
            sim_result["pc_htf_pump_power"],
            sim_result["field_htf_pump_power"],
            sim_result["field_collector_tracking_power"],
            sim_result["pc_startup_thermal_power"],
            sim_result["field_piping_thermal_loss"],
            sim_result["receiver_thermal_loss"],
            hour_index=hour_index,
        )

    else:
        raise ValueError(f"Invalid optim_mode: {optim_mode}")
