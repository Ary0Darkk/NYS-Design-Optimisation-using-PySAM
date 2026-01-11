import optuna
from prefect.logging import get_run_logger
from optimisation.rl_optimiser import train_rl


def objective(
    trial,
    override,
):
    logger = get_run_logger()
    # Optuna will pick new values for these for every trial
    tuning_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [16, 32, 64, 128]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "ent_coef": trial.suggest_float("ent_coef", 0.0001, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
    }

    # Call your training function with these new values
    # We modify your train_rl to accept these (see Step 2)
    _, best_reward, _ = train_rl(
        override=override,
        static_overrides={},
        hour_index=1,
        tuned_hyperparams=tuning_params,  # Pass the suggestions here
        is_tuning=True,
        trial=trial,  # Pass the trial object for pruning
    )

    logger.info(f"Best reward : {best_reward}")
    # Optuna wants to MAXIMIZE the reward
    return best_reward


def run_rl_study(override):
    logger = get_run_logger()
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
    study.optimize(lambda trial: objective(trial, override), n_trials=50)

    logger.info("Finished optimisation!")
