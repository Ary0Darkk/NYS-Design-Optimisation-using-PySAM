import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO


class MixedOptimizationEnv(gym.Env):
    """
    Custom Environment for Mixed-Integer Optimization.
    Goal: Minimize f(i, c) = (i - 7)^2 + (c - 2.5)^2
    Optimal solution: i = 7, c = 2.5
    """

    def __init__(self):
        super(MixedOptimizationEnv, self).__init__()

        # We define a Box space for both.
        # index 0: The integer parameter (will be rounded)
        # index 1: The float parameter
        low = np.array([0.0, -5.0], dtype=np.float32)
        high = np.array([10.0, 5.0], dtype=np.float32)

        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        # Observation space (the agent's current parameters)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(2,), dtype=np.float32
        )

        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.max_steps = 20
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start at a random point within the bounds
        self.state = self.observation_space.sample()
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        # --- MIXED SPACE LOGIC ---
        # 1. Handle Integer: Round the first element to the nearest whole number
        int_param = int(np.round(action[0]))
        # 2. Handle Float: Use the second element as is
        float_param = action[1]

        # Update state for observation
        self.state = np.array([float(int_param), float_param], dtype=np.float32)

        # --- OPTIMIZATION TARGET ---
        # We want to minimize (i-7)^2 + (c-2.5)^2
        # Reward is the negative of the cost function
        target_i = 7
        target_c = 2.5
        cost = (int_param - target_i) ** 2 + (float_param - target_c) ** 2
        reward = -cost

        # Termination conditions
        terminated = bool(cost < 0.01)
        truncated = bool(self.current_step >= self.max_steps)

        return self.state, reward, terminated, truncated, {}


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Initialize Env
    env = MixedOptimizationEnv()

    # 2. Setup Agent (PPO handles continuous Box spaces well)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

    # 3. Train
    print("Training started...")
    model.learn(total_timesteps=20000)
    print("Training finished.")

    # 4. Test the Agent
    print("\n--- Testing the Agent ---")
    obs, _ = env.reset()
    for _ in range(5):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Print results clearly
        actual_int = int(np.round(obs[0]))
        actual_float = obs[1]
        print(
            f"Action taken -> Integer: {actual_int}, Float: {actual_float:.4f} | Reward: {reward:.4f}"
        )

        if terminated or truncated:
            break
