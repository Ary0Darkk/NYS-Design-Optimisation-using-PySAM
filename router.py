# file to route design optimisation and operational one
from typing import Optional

from optimisation import *
from optimisation.rl_optimiser.rl_tuner import run_rl_study

from config import CONFIG
import mlflow
import dagshub

import logging

logger = logging.getLogger("NYS_Optimisation")


def optimisation_mode() -> str | dict[str, list[float]]:
    override = None
    if CONFIG["route"] == "design":
        override = "design"
    elif CONFIG["route"] == "operational":
        override = "operational"
    elif CONFIG["route"] == "des-operational":
        override = "design_operational"
    else:
        print(f"{CONFIG['route']} : Not found! ")

    logger.info(f"{override} route taken!")

    return override


def call_optimiser(
    override: dict[str, list[float]],
    optim_mode: str,
    is_nested: bool,
    target_hour: int,
    static_overrides: Optional[dict[str, float]] = None,
):
    # FIXME : try and catch is not working as expected,
    # look at keyboard interupt working

    if static_overrides is None:
        static_overrides = {}

    # initilise return variables
    x_opt = None
    f_val = None

    try:
        # optimisation
        opt_type = CONFIG.get("optimiser")

        if opt_type == "fmincon":
            x_opt, f_val = run_fmincon_optimisation()
        elif opt_type == "ga":
            x_opt, f_val = run_ga_optimisation()
        elif opt_type == "pygad_ga":
            x_opt, f_val, _ = run_pyga_optimisation()
        elif opt_type == "nlopt":
            x_opt, f_val, _ = run_nlopt()
        elif opt_type == "scipy_min":
            x_opt, f_val, _ = run_scipy_minimise()
        elif opt_type == "deap_ga":
            x_opt, f_val, o_metrices = run_deap_ga_optimisation(
                override=override,
                optim_mode=optim_mode,
                static_overrides=static_overrides,
                is_nested=is_nested,
                curr_hour=target_hour,
            )
        elif opt_type == "rl_optim":
            x_opt, f_val, o_metrices = train_rl(
                override=override,
                optim_mode=optim_mode,
                static_overrides=static_overrides,
                is_nested=is_nested,
                hour_index=target_hour,
            )
        else:
            print(f"{opt_type} : Not a valid optimiser name in CONFIG")

        # only print if the variables were successfully set
        if x_opt is not None:
            logger.info(f"x_opt : {x_opt}")
            logger.info(f"f_val : {f_val}")

    except KeyboardInterrupt:
        logger.warning("\n\nOptimization interrupted by user. Stopping...\n")
    # except Exception as e:
    #     print("Unexpected error :", e)

    return x_opt, f_val, o_metrices


def call_tuner(override: dict[str, list[float]]):
    # call run study
    run_rl_study(
        override=override,
    )


def run_hourly_optimisation(
    override: dict[str, list[float]],
    optim_mode: str,
    is_nested: bool,
    static_overrides: Optional[dict[str, float]] = None,
):
    results = {}

    for hour in range(1, 8761):
        print(f"\n{'-' * 40}")
        print(f"Optimising for hour {hour}")
        print(f"{'-' * 40}")

        best_x, best_f, _ = call_optimiser(
            override=override,
            optim_mode=optim_mode,
            static_overrides=static_overrides,
            is_nested=is_nested,
            target_hour=hour,
        )

        results[hour] = {
            "best_solution": best_x,
            "best_fitness": best_f,
        }

    return results


def run_router():
    # database setup
    mlflow.set_tracking_uri(
        "https://dagshub.com/aryanvj787/NYS-Design-Optimisation-using-PySAM.mlflow"
    )
    dagshub.init(
        repo_owner="aryanvj787",
        repo_name="NYS-Design-Optimisation-using-PySAM",
        mlflow=True,
    )

    if CONFIG.get("is_tuning", False):
        # set experiment name
        mlflow.set_experiment("rl-tuning")

        call_tuner(override=CONFIG["design"])
    else:
        # optimisation
        opt_type = CONFIG.get("optimiser")

        # set experiment name
        mlflow.set_experiment(f"{opt_type}-optimisation")

        # design and operational optim logic
        optimals = None

        override = optimisation_mode()

        # only perfoms single optim

        if override == "design":
            logger.info(f"{CONFIG['route']} optimisation started !")
            # print(f"Optimisation of : {override}")
            call_optimiser(
                override=CONFIG["design"],
                target_hour=1,
                optim_mode="design",
                is_nested=False,
            )

        elif override == "operational":
            is_nested = True

            with mlflow.start_run(run_name="Operational optimisation"):
                logger.info(f"{CONFIG['route']} optimisation started !")
                run_hourly_optimisation(
                    override=CONFIG["operational"],
                    optim_mode="operational",
                    is_nested=is_nested,
                )
        # perform both optim in sequence
        elif override == "design_operational":
            is_nested = True  # informs mlflow for multi-step run
            # 1. Start a Parent Run to group everything
            with mlflow.start_run(run_name="Sequential des-operational"):
                logger.info(
                    "Going to begin Design plus Operational optimisation sequentially!\n\n"
                )
                logger.info("Design optimisation started !")
                # print(f"Optimisation of : {override}")
                optimals = call_optimiser(
                    override=CONFIG["design"],
                    optim_mode="design",
                    target_hour=1,
                    is_nested=is_nested,
                )

                if optimals is not None:
                    design_dict = dict(zip(CONFIG["design"]["overrides"], optimals[0]))

                    logger.info("Operational optimisation started !")
                    # print(f"Optimisation of : {override}")
                    run_hourly_optimisation(
                        override=CONFIG["operational"],
                        optim_mode="operational",
                        static_overrides=design_dict,
                        is_nested=is_nested,
                    )

                else:
                    logger.info("Design optimisation is not performed yet!")
        else:
            logger.info(f"{opt_type} is not a valid optimiser!")
