# file to route design optimisation and operational one
from typing import Optional

from optimisation import *

from config import CONFIG
import mlflow
import dagshub

from prefect import task
from prefect.logging import get_run_logger


@task()
def optimisation_mode() -> str | dict[str, list[float]]:
    logger = get_run_logger()

    override = None
    if CONFIG["route"] == "design":
        override = CONFIG["design"]
    elif CONFIG["route"] == "operational":
        override = CONFIG["operational_overrides"]
    elif CONFIG["route"] == "des-operational":
        override = "design_operational"
    else:
        print(f"{CONFIG['route']} : Not found! ")

    logger.debug(f"{override} route taken!")

    return override


@task()
def call_optimiser(
    override: dict[str, list[float]],
    is_nested: bool,
    target_hour: int,
    static_overrides: Optional[dict[str, float]] = None,
):
    # FIXME : try and catch is not working as expected,
    # look at keyboard interupt working
    logger = get_run_logger()

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
                static_overrides=static_overrides,
                is_nested=is_nested,
                curr_hour=target_hour,
            )
        elif opt_type == "rl_optim":
            x_opt, f_val, o_metrices = train_rl(
                override=override,
                static_overrides=static_overrides,
                is_nested=is_nested,
                hour_index=target_hour,
            )
        else:
            print(f"{opt_type} : Not a valid optimiser name in CONFIG")

        logger.debug(f"{opt_type} started!")
        # only print if the variables were successfully set
        if x_opt is not None:
            logger.info(f"x_opt : {x_opt}")
            logger.info(f"f_val : {f_val}")

    except KeyboardInterrupt:
        logger.warning("\n\nOptimization interrupted by user. Stopping...\n")
    # except Exception as e:
    #     print("Unexpected error :", e)

    return x_opt, f_val, o_metrices


@task()
def run_hourly_optimisation(
    override: dict[str, list[float]],
    is_nested: bool,
    static_overrides: Optional[dict[str, float]] = None,
):
    results = {}

    for hour in range(1, 8761):
        print(f"\n{'-' * 20}")
        print(f"Optimising for hour {hour}")
        print(f"{'-' * 20}")

        best_x, best_f, _ = call_optimiser(
            override=override,
            static_overrides=static_overrides,
            is_nested=is_nested,
            target_hour=hour,
        )

        results[hour] = {
            "best_solution": best_x,
            "best_fitness": best_f,
        }

    return results


@task()
def run_router():
    logger = get_run_logger()
    # database setup
    mlflow.set_tracking_uri(
        "https://dagshub.com/aryanvj787/NYS-Design-Optimisation-using-PySAM.mlflow"
    )
    dagshub.init(
        repo_owner="aryanvj787",
        repo_name="NYS-Design-Optimisation-using-PySAM",
        mlflow=True,
    )

    # optimisation
    opt_type = CONFIG.get("optimiser")

    # set experiment name
    mlflow.set_experiment(f"{opt_type}-optimisation")

    # design and operational optim logic
    optimals = None

    override = optimisation_mode()

    # only perfoms single optim
    if override != "design_operational":
        logger.debug(f"{CONFIG['route']} optimisation started !")
        # print(f"Optimisation of : {override}")
        run_hourly_optimisation(override, is_nested=False)
    # perform both optim in sequence
    else:
        is_nested = True  # informs mlflow for multi-step run
        # 1. Start a Parent Run to group everything
        with mlflow.start_run(run_name="Sequential des-operational"):
            logger.debug(
                "Going to begin Design plus Operational optimisation sequentially!\n\n"
            )
            logger.debug(f"Design optimisation started !")
            # print(f"Optimisation of : {override}")
            optimals = run_hourly_optimisation(
                override=CONFIG["design"], is_nested=is_nested
            )

            if optimals is not None:
                design_dict = dict(zip(CONFIG["design"]["overrides"], optimals[0]))

                logger.debug(f"Operational optimisation started !")
                # print(f"Optimisation of : {override}")
                run_hourly_optimisation(
                    override=CONFIG["operational"],
                    static_overrides=design_dict,
                    is_nested=is_nested,
                )

            else:
                logger.warning("Design optimisation is not performed yet!")
