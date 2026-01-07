from pathlib import Path
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from prefect import task

import multiprocessing

from .rl_env import SolarMixedOptimisationEnv
from config import CONFIG


def make_env(
    var_names,
    var_types,
    lb,
    ub,
    static_overrides,
    hour_index,
    max_steps,
):
    def _init():
        return SolarMixedOptimisationEnv(
            var_names=var_names,
            var_types=var_types,
            lb=lb,
            ub=ub,
            static_overrides=static_overrides,
            hour_index=hour_index,
            max_steps=max_steps,
        )

    return _init


@task()
def train_rl(
    override,
    static_overrides,
    hour_index,
    is_nested=False,
):
    var_names = override["overrides"]
    var_types = override["types"]
    lb, ub = override["lb"], override["ub"]

    run_name = f"RL_hour_{hour_index}"

    if mlflow.active_run() and not is_nested:
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name, nested=is_nested):
        mlflow.set_tag("optimiser", "PPO")
        mlflow.set_tag("hour", hour_index)

        num_envs = min(4, multiprocessing.cpu_count())  # use physical cores

        env = SubprocVecEnv(
            [
                make_env(
                    var_names,
                    var_types,
                    lb,
                    ub,
                    static_overrides,
                    hour_index,
                    CONFIG["rl_max_steps"],
                )
                for _ in range(num_envs)
            ]
        )

        # ---- Checkpoint directory ----
        ckpt_dir = Path("checkpoints") / "rl" / f"hour_{hour_index}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_path = ckpt_dir / "ppo_latest.zip"

        # ---- Resume or fresh ----
        if model_path.exists() and CONFIG.get("resume_from_checkpoint", False):
            model = PPO.load(model_path, env=env)
        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=20,
                batch_size=10,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                device="cpu",  # available
                verbose=1,
            )

        checkpoint_cb = CheckpointCallback(
            save_freq=CONFIG.get("rl_checkpoint_freq"),
            save_path=str(ckpt_dir),
            name_prefix="ppo",
        )

        total_timesteps = CONFIG.get("rl_timesteps", 2)

        print("PPO device:", model.policy.device)
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_cb,
        )

        # predicts the best params using determinitic policy
        obs = env.reset()
        best_reward = -float("inf")
        best_params = None

        for _ in range(CONFIG.get("rl_eval_steps")):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            max_idx = rewards.argmax()

            if rewards[max_idx] > best_reward:
                best_reward = rewards[max_idx]
                best_params = infos[max_idx]["overrides"]

            if dones.any():
                obs = env.reset()

        print("\n" + "-" * 40)
        print(f"RL Optimal solution (hour {hour_index})")
        print("-" * 40)
        for k, v in best_params.items():
            print(f"{k:20}: {v}")
        print("-" * 40)
        print(f"Best reward: {best_reward:.6f}")
        print("-" * 40)

        mlflow.log_metrics({"best_reward": best_reward})
        for k, v in best_params.items():
            mlflow.log_param(f"opt_{k}", v)

        model.save(model_path)

        # ---- Log artifacts ----
        mlflow.log_artifact(str(model_path))
        mlflow.log_params(
            {
                "timesteps": total_timesteps,
                "learning_rate": model.learning_rate,
            }
        )

        return best_params, best_reward, model
