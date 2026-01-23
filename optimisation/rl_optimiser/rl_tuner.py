import optuna
import logging
from .ppo_rl_training import train_rl

logger = logging.getLogger("NYS_Optimisation")


def objective(
    trial,
    override,
):
    # Optuna will pick new values for these for every trial
    # define your hyperparamters range
    tuning_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [16, 32, 64, 128]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "ent_coef": trial.suggest_float("ent_coef", 0.0001, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
    }

    # calls training function
    _, best_reward, _ = train_rl(
        override=override,
        static_overrides={},
        hour_index=1,
        tuned_hyperparams=tuning_params,
        is_tuning=True,
        trial=trial,  # pass the trial object for pruning
    )

    logger.info(f"Best reward : {best_reward}")
    # Optuna wants to MAXIMIZE the reward
    return best_reward


def run_rl_study(override):
    # This DB allows your 2 PCs to share the same tuning work
    # storage = "postgresql://user:pass@your-main-pc-ip:5432/optuna_db"

    logger.info("Started study for hyperparameter tuning!")
    study = optuna.create_study(
        study_name="solar_rl_tuning",
        # storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),  # Kills bad runs early
    )
    logger.info("Now optimising it!")
    study.optimize(
        lambda trial: objective(trial, override), n_trials=2, show_progress_bar=True
    )

    logger.info("Finished optimisation!")
