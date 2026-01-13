import optuna
from ga_optimiser import run_deap_ga_optimisation


def objective(trial, override, static_overrides, hour_index):
    # 1. Define GA Search Space
    tuning_params = {
        "n_pop": trial.suggest_int("n_pop", 20, 200, step=10),
        "cxpb": trial.suggest_float("cxpb", 0.1, 0.9),
        "mutpb": trial.suggest_float("mutpb", 0.01, 0.5),
        "n_gen": trial.suggest_int("n_gen", 10, 100),
    }

    # 2. Run the GA Task
    _, best_reward = run_deap_ga_optimisation.fn(
        override=override,
        static_overrides=static_overrides,
        hour_index=hour_index,
        tuned_params=tuning_params,
    )

    return best_reward


def run_ga_study(override, static_overrides, hour_index):
    study = optuna.create_study(
        study_name="solar_ga_tuning",
        # storage=CONFIG["tuning"]["db_url"], # Same DB as RL
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, override, static_overrides, hour_index),
        n_trials=30,
    )
