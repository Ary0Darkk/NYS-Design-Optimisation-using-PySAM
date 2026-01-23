import gymnasium as gym
from gymnasium import spaces
import numpy as np

from simulation import run_simulation
from objective_functions import objective_function
from config import CONFIG


class SolarMixedOptimisationEnv(gym.Env):
    """
    RL environment compatible with GA objective.
    Action = continuous, internally mapped to int/float.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        var_names,
        var_types,
        lb,
        ub,
        static_overrides,
        hour_index,
        max_steps,
        optim_mode,
    ):
        super().__init__()

        self.var_names = var_names
        self.var_types = var_types
        self.lb = np.array(lb, dtype=np.float32)
        self.ub = np.array(ub, dtype=np.float32)
        self.static_overrides = static_overrides
        self.hour_index = hour_index
        self.optim_mode = optim_mode

        self.action_space = spaces.Box(
            low=self.lb,
            high=self.ub,
            dtype=np.float32,
        )

        self.observation_space = self.action_space

        self.max_steps = max_steps
        self.current_step = 0
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self.np_random.uniform(low=self.lb, high=self.ub).astype(
            np.float32
        )  # initialise from some random value for exploration
        # self.state = np.zeros_like(self.action_space.low)
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        # ---- Mixed variable handling (same as GA) ----
        overrides_dyn = {}
        for i, name in enumerate(self.var_names):
            val = action[i]

            overrides_dyn[name] = float(
                val
            )  # type casting self.var_types[i] -> int or float then (val) type cast val into that type

        self.last_overrides = overrides_dyn
        self.static_overrides = {
            k: float(v) for k, v in self.static_overrides.items()
        }  # pysam can't deal with numpy floats

        final_overrides = {**overrides_dyn, **self.static_overrides}

        # ---- Run simulation ----

        sim_result = run_simulation(final_overrides)

        if self.optim_mode == "design":
            try:
                obj = sim_result["annual_energy"]
            except KeyError:
                # This will print the ACTUAL keys being returned by the cached task
                print(
                    f"CRITICAL: 'annual_energy' missing. Available keys: {list(sim_result.keys())}"
                )
                obj = 0  # Return a penalty score instead of crashing
        elif self.optim_mode == "operational":
            obj = objective_function(
                sim_result["hourly_energy"],
                sim_result["pc_htf_pump_power"],
                sim_result["field_htf_pump_power"],
                sim_result["field_collector_tracking_power"],
                sim_result["pc_startup_thermal_power"],
                sim_result["field_piping_thermal_loss"],
                sim_result["receiver_thermal_loss"],
                hour_index=self.hour_index,
            )
        else:
            print(f"{self.optim_mode} is invalid!")

        reward = np.float32(obj)

        # TODO : Add your constraints here, refer to below example
        # Check Constraints (e.g., Battery SoC)
        # battery_soc = next_state[0]

        # if battery_soc < 0.2:
        #     # Apply a heavy penalty
        #     penalty = -50.0
        #     reward += penalty

        # Optional: End the episode early if the constraint is critical
        # done = True

        self.state = np.array(
            list(overrides_dyn.values()), dtype=np.float32
        )  # setting dtype to object maintains the exact precision
        # print(self.state.dtype)

        # ---- max steps limit ----
        terminated = False
        truncated = self.current_step >= self.max_steps

        return self.state, reward, terminated, truncated, {"overrides": overrides_dyn}
