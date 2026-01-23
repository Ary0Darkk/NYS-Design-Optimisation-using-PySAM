from pathlib import Path
import numpy as np
import tabulate as tb
import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import pandas as pd

import multiprocessing

from .rl_env import SolarMixedOptimisationEnv
from .rl_util import TrialEvalCallback
from config import CONFIG

import logging

logger = logging.getLogger("NYS_Optimisation")


# creates envs
def make_env(
    var_names,
    var_types,
    lb,
    ub,
    static_overrides,
    hour_index,
    max_steps,
    optim_mode,
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
            optim_mode=optim_mode,
        )

    return _init


def train_rl(
    override,
    optim_mode,
    static_overrides,
    hour_index,
    is_nested=False,
    tuned_hyperparams=None,  # dict from Optuna
    is_tuning=False,  # flag to change behavior
    trial=None,  # Optuna Trial object
):
    try:
        if optim_mode == "design":
            run_name = "RL_Design_optimisation"
        else:
            run_name = f"RL_hour_{hour_index}"

        if mlflow.active_run() and not is_nested:
            mlflow.end_run()

        with mlflow.start_run(run_name=run_name, nested=is_nested):
            mlflow.set_tag("Author", CONFIG["author"])
            mlflow.set_tag("optimiser", "PPO")
            mlflow.log_artifact("config.py")
            mlflow.set_tag("hour", hour_index)

            var_names = override["overrides"]
            var_types = override["types"]
            lb, ub = override["lb"], override["ub"]

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
                        optim_mode,
                    )
                    for _ in range(num_envs)
                ]
            )

            # ---- Checkpoint directory ----
            ckpt_dir = Path("checkpoints") / "rl" / f"hour_{hour_index}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            model_path = ckpt_dir / "ppo_latest.zip"

            # ---- Handle Hyperparameters ----
            # We prioritize Optuna, then fallback to CONFIG, then fallback to SB3 defaults
            hp = tuned_hyperparams or {}

            def get_val(key, default):
                # checks Optuna trial, global CONFIG, Hardcoded fallback
                return hp.get(key, CONFIG.get(f"rl_{key}", default))

            # ---- Resume or fresh ----
            if model_path.exists() and CONFIG.get("resume_from_checkpoint", False):
                model = PPO.load(model_path, env=env)
            else:
                model = PPO(
                    "MlpPolicy",
                    env,
                    learning_rate=get_val("learning_rate", 3e-4),
                    n_steps=get_val("n_steps", 20),
                    batch_size=get_val("batch_size", 10),
                    ent_coef=get_val("ent_coef", 0.01),
                    gamma=get_val("gamma", 0.99),
                    n_epochs=10,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    device="cpu",  # available
                    verbose=1,
                    seed=CONFIG.get("random_seed", 21),
                )

            checkpoint_cb = CheckpointCallback(
                save_freq=CONFIG.get("rl_checkpoint_freq", 10),
                save_path=str(ckpt_dir),
                name_prefix="ppo",
            )

            # ---- Add Pruning Callback ----
            callbacks = [checkpoint_cb]
            if is_tuning and trial:
                # This allows Optuna to kill bad trials early
                eval_env = make_env(
                    var_names,
                    var_types,
                    lb,
                    ub,
                    static_overrides,
                    hour_index,
                    CONFIG["rl_max_steps"],
                )()
                # This is the custom class we discussed that reports back to 'trial'
                tuning_cb = TrialEvalCallback(eval_env, trial, eval_freq=20)
                callbacks.append(tuning_cb)

            total_timesteps = CONFIG.get("rl_timesteps", 2)

            logger.info(f"PPO device: {model.policy.device}")
            model.learn(
                total_timesteps=total_timesteps,
                callback=checkpoint_cb,
            )

            # predicts the best params using determinitic policy
            obs = env.reset()
            best_reward = -np.float32("inf")
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

            res_dict = {}
            header_line = "-" * 40
            if optim_mode == "design":
                for k, v in best_params.items():
                    res_dict[k] = v
                res_dict["best_reward"] = best_reward

                #    formats output
                res_table = tb.tabulate(res_dict.items(), tablefmt="grid")
                logger.info(
                    f"\n{header_line}\n"
                    f"RL DESIGN OPTIMAL SOLUTIONS\n"
                    f"{header_line}\n"
                    f"Final Best Results\n"
                    f"{res_table}"
                )
            else:
                res_dict["hour"] = hour_index
                for k, v in best_params.items():
                    res_dict[k] = v
                res_dict["best_reward"] = best_reward

                # formats output
                res_table = tb.tabulate(res_dict.items(), tablefmt="grid")
                logger.info(
                    f"\n{header_line}\n"
                    f"RL Optimal solution (hour {hour_index})\n"
                    f"{header_line}\n"
                    f"Final Best Results\n"
                    f"{res_table}"
                )

            result_logbook = pd.DataFrame([res_dict])
            result_logbook.index = result_logbook.index + 1
            result_logbook.index.name = "serial"

            if optim_mode == "design":
                file_name = Path("results/RL_design_results.csv")
            else:
                file_name = Path("results/RL_operational_results.csv")
            file_name.parent.mkdir(exist_ok=True)

            file_exists = file_name.exists()
            result_logbook.to_csv(file_name, mode="a", header=not file_exists)

            mlflow.log_metric("best_reward", best_reward)
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

            return best_params.values(), best_reward, total_timesteps
    except KeyboardInterrupt:
        print("Interrupted by User!\nStopping...")

    finally:
        # close worker pool
        print("Closing worker pool...")
        env.close()
