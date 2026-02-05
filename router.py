# file to route design optimisation and operational one
from typing import Optional
import os

from optimisation import *
from optimisation.rl_optimiser.rl_tuner import run_rl_study

from config import CONFIG
import mlflow
from multiprocessing import Pool, cpu_count

import logging
from optimisation.rl_optimiser.ppo_rl_training import make_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from utilities.mlflow_init import initialize_mlflow
from utilities.hour_sampling import build_operating_hours_from_month_day

logger = logging.getLogger("NYS_Optimisation")


def optimisation_mode() -> str:
    route = None
    if CONFIG["route"] == "design":
        route = "design"
    elif CONFIG["route"] == "operational":
        route = "operational"
    elif CONFIG["route"] == "design_operational":
        route = "design_operational"
    else:
        print(f"{CONFIG['route']} : Not found! ")

    logger.info(f"{route} route taken!")

    return route


def call_optimiser(
    override: dict[str, list[float]],
    optim_mode: str,
    is_nested: bool,
    target_hour: int,
    pool,
    rec=None,
    static_overrides: Optional[dict[str, float]] = None,
):
    # FIXME : try and catch is not working as expected,
    # look at keyboard interupt working

    if static_overrides is None:
        static_overrides = {}

    # initilise return variables
    x_opt = None
    f_val = None
    o_metrices = None

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
                pool=pool,
                rec=rec,
            )
        elif opt_type == "rl_optim":
            x_opt, f_val, o_metrices = train_rl(
                override=override,
                optim_mode=optim_mode,
                static_overrides=static_overrides,
                is_nested=is_nested,
                hour_index=target_hour,
                env=pool,
                rec=rec,
            )
        else:
            print(f"{opt_type} : Not a valid optimiser name in CONFIG")

        # only print if the variables were successfully set
        if x_opt is not None:
            logger.info(f"x_opt : {x_opt}")
            logger.info(f"f_val : {f_val}")

    except KeyboardInterrupt:
        logger.warning("\n\nOptimization interrupted by user. Stopping...\n")
    except Exception as e:
        print("Unexpected error :", e)

    return x_opt, f_val, o_metrices


def call_tuner(override: dict[str, list[float]]):
    # call run study
    run_rl_study(
        override=override,
    )


# runs hourly optimisation
def run_hourly_optimisation(
    override: dict[str, list[float]],
    optim_mode: str,
    is_nested: bool,
    pool,
    static_overrides: Optional[dict[str, float]] = None,
):
    results = {}
    operating_records = build_operating_hours_from_month_day(
        CONFIG["USER_DEFINED_DAYS"]
    )
    try:
        for rec in operating_records:
            hour = rec["sam_hour"]

            logger.info(
                f"\n{'-' * 30}\n"
                f"Season : {rec['season']}\n"
                f"Date   : {rec['day']:02d}-{rec['month']:02d}-2020\n"
                f"Hour   : {rec['hour_of_day']:02d}:00–{rec['hour_of_day'] + 1:02d}:00\n"
                f"SAM hr : {hour}\n"
                f"{'-' * 30}"
            )

            best_x, best_f, _ = call_optimiser(
                override=override,
                optim_mode=optim_mode,
                static_overrides=static_overrides,
                is_nested=is_nested,
                target_hour=hour,
                pool=pool,
                rec=rec,
            )

            results[hour] = {"best_solution": best_x, "best_fitness": best_f}
    except KeyboardInterrupt:
        logger.warning(f"\nStopped at hour {hour} by user.")
        # We do NOT close the pool here; we let 'finally' or the caller handle it
        # so we don't accidentally close a pool that might be needed for cleanup
        raise

    return results


def run_router():
    # database setup
    # mlflow.set_tracking_uri(
    #     "https://dagshub.com/aryanvj787/NYS-Design-Optimisation-using-PySAM.mlflow"
    # )
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    initialize_mlflow(
        repo_owner="aryanvj787", repo_name="NYS-Design-Optimisation-using-PySAM"
    )
    # dagshub.init(
    #     repo_owner="aryanvj787",
    #     repo_name="NYS-Design-Optimisation-using-PySAM",
    #     mlflow=True,
    # )

    if CONFIG.get("is_tuning", False):
        # set experiment name
        mlflow.set_experiment("rl-tuning")

        call_tuner(override=CONFIG["design"])
    else:
        rl_env = None
        # optimisation
        opt_type = CONFIG.get("optimiser")

        # set experiment name
        mlflow.set_experiment(f"{opt_type}-optimisation")
        logger.info(f"Starting {opt_type} optimiser...")

        # design and operational optim logic

        optimals = None  # stores static override

        # only perfoms single optim

        route = optimisation_mode()

        # get cpu count from configs
        config_cpus = CONFIG.get("num_cores")

        # if config is empty/None
        if not config_cpus:
            try:
                # works on Linux and respects PBS/Cgroups limits
                default_cpus = len(os.sched_getaffinity(0))
            except AttributeError:
                # Fallback for Windows (your laptop)
                default_cpus = os.cpu_count()

            n_cores = default_cpus
        else:
            n_cores = int(config_cpus)

        # --------- design route ------------------------------------------------
        if route == "design":
            if opt_type == "deap_ga":
                # initialize the Pool ONCE at the start
                logger.info(f"Initializing persistent pool with {n_cores} cores.")
                logger.info(f"{route} optimisation started !")
                global_pool = Pool(processes=n_cores)
                try:
                    # print(f"Optimisation of : {override}")
                    call_optimiser(
                        override=CONFIG[route],
                        target_hour=1,
                        optim_mode=route,
                        is_nested=False,
                        pool=global_pool,
                    )
                except KeyboardInterrupt:
                    logger.warning(
                        "\nParent process received interrupt. Terminating workers..."
                    )
                    global_pool.terminate()  # Instantly kill all workers
                    global_pool.join()
                    logger.info("Pool terminated successfully.")
                    raise  # Re-raise to stop the entire script
                finally:
                    global_pool.close()
                    global_pool.join()
                    logger.info(f"Closed {n_cores} workers pool!")

            elif opt_type == "rl_optim":
                logger.info(f"{route} optimisation started !")
                logger.info(f"Launching {n_cores} persistent RL worker environments...")
                try:
                    # Initialize env once with hour 1
                    override = CONFIG[route]
                    rl_env = SubprocVecEnv(
                        [
                            make_env(
                                override["overrides"],
                                override["types"],
                                override["lb"],
                                override["ub"],
                                {},
                                1,
                                CONFIG["rl_max_steps"],
                                optim_mode=route,
                                seed=CONFIG.get["random_seed"],
                            )
                            for _ in range(n_cores)
                        ]
                    )
                    # calls optimiser
                    call_optimiser(
                        override=override,
                        target_hour=1,
                        optim_mode=route,
                        is_nested=False,
                        pool=rl_env,
                    )
                except KeyboardInterrupt:
                    logger.warning("User interrupted the process.")

                finally:
                    # catch-all cleanup
                    if rl_env is not None:
                        logger.info("Terminating RL environments...")
                        rl_env.close()
                        logger.info("RL environments closed.")

            else:
                logger.info(f"{opt_type} is not a valid optimiser!")

        # --------- operational ---------------------------------------------------
        elif route == "operational":
            is_nested = True

            with mlflow.start_run(run_name="Operational optimisation"):
                logger.info(f"{route} optimisation started !")
                if opt_type == "deap_ga":
                    # initialize the Pool ONCE at the start
                    logger.info(f"Initializing persistent pool with {n_cores} cores.")
                    global_pool = Pool(processes=n_cores)
                    try:
                        # print(f"Optimisation of : {override}")
                        run_hourly_optimisation(
                            override=CONFIG[route],
                            optim_mode="operational",
                            is_nested=is_nested,
                            pool=global_pool,
                        )
                    except KeyboardInterrupt:
                        logger.warning(
                            "\nParent process received interrupt. Terminating workers..."
                        )
                        global_pool.terminate()  # Instantly kill all workers
                        global_pool.join()
                        logger.info("Pool terminated successfully.")
                        raise  # Re-raise to stop the entire script
                    finally:
                        global_pool.close()
                        global_pool.join()
                        logger.info(f"Closed {n_cores} workers pool!")

                elif opt_type == "rl_optim":
                    logger.info(
                        f"Launching {n_cores} persistent RL worker environments..."
                    )
                    try:
                        # Initialize env once with hour 1
                        override = CONFIG[route]
                        rl_env = SubprocVecEnv(
                            [
                                make_env(
                                    override["overrides"],
                                    override["types"],
                                    override["lb"],
                                    override["ub"],
                                    {},
                                    1,
                                    CONFIG["rl_max_steps"],
                                    optim_mode="operational",
                                    seed=CONFIG.get["random_seed"],
                                )
                                for _ in range(n_cores)
                            ]
                        )
                        # print(f"Optimisation of : {override}")
                        run_hourly_optimisation(
                            override=override,
                            optim_mode="operational",
                            is_nested=is_nested,
                            pool=rl_env,
                        )
                    except KeyboardInterrupt:
                        logger.warning("User interrupted the process.")

                    finally:
                        # catch-all cleanup
                        if rl_env is not None:
                            logger.info("Terminating RL environments...")
                            rl_env.close()
                            logger.info("RL environments closed.")

                else:
                    logger.info(f"{opt_type} is not a valid optimiser!")

        # ------------ design + operational -----------------------------------------------
        # perform both optim in sequence
        elif route == "design_operational":
            is_nested = True  # informs mlflow for multi-step run
            # start a parent run to group everything
            with mlflow.start_run(run_name="Sequential des-operational"):
                logger.info(
                    "Going to begin Design plus Operational optimisation sequentially!\n\n"
                )
                if opt_type == "deap_ga":
                    logger.info("Design optimisation started !")
                    # initialize the Pool ONCE at the start
                    logger.info(f"Initializing persistent pool with {n_cores} cores.")
                    global_pool = Pool(processes=n_cores)

                    try:
                        # print(f"Optimisation of : {override}")
                        optimals = call_optimiser(
                            override=CONFIG["design"],
                            optim_mode="design",
                            target_hour=1,
                            is_nested=is_nested,
                            pool=global_pool,
                        )

                        if optimals[0] is not None:
                            # Force every value to be a native Python float
                            static_override_dict = {
                                name: float(val)
                                for name, val in zip(
                                    CONFIG["design"]["overrides"], optimals[0]
                                )
                            }

                            logger.info("Operational optimisation started !")
                            # print(f"Optimisation of : {override}")
                            run_hourly_optimisation(
                                override=CONFIG["operational"],
                                optim_mode="operational",
                                static_overrides=static_override_dict,
                                is_nested=is_nested,
                                pool=global_pool,
                            )
                        else:
                            logger.info("Design optimisation is not performed yet!")
                    except KeyboardInterrupt:
                        logger.warning(
                            "\nParent process received interrupt. Terminating workers..."
                        )
                        global_pool.terminate()  # Instantly kill all workers
                        global_pool.join()
                        logger.info("Pool terminated successfully.")
                        raise  # Re-raise to stop the entire script
                    finally:
                        global_pool.close()
                        global_pool.join()
                        logger.info(f"Closed {n_cores} workers pool!")

                    logger.info("Completed optimisation")

                elif opt_type == "rl_optim":
                    logger.info("Design optimisation started !")
                    logger.info(
                        f"Launching {n_cores} persistent RL worker environments..."
                    )
                    try:
                        # Initialize env once with hour 1
                        override = CONFIG["design"]
                        rl_env = SubprocVecEnv(
                            [
                                make_env(
                                    override["overrides"],
                                    override["types"],
                                    override["lb"],
                                    override["ub"],
                                    {},
                                    1,
                                    CONFIG["rl_max_steps"],
                                    optim_mode="operational",
                                    seed=CONFIG.get["random_seed"],
                                )
                                for _ in range(n_cores)
                            ]
                        )
                        # print(f"Optimisation of : {override}")
                        optimals = call_optimiser(
                            override=override,
                            optim_mode="design",
                            target_hour=1,
                            is_nested=is_nested,
                            pool=rl_env,
                        )

                        if optimals[0] is not None:
                            # Force every value to be a native Python float
                            static_override_dict = {
                                name: float(val)
                                for name, val in zip(
                                    CONFIG["design"]["overrides"], optimals[0]
                                )
                            }
                            # Initialize env once with hour 1
                            override = CONFIG["operational"]
                            rl_env = SubprocVecEnv(
                                [
                                    make_env(
                                        override["overrides"],
                                        override["types"],
                                        override["lb"],
                                        override["ub"],
                                        static_override_dict,
                                        1,
                                        CONFIG["rl_max_steps"],
                                        optim_mode="operational",
                                    )
                                    for _ in range(n_cores)
                                ]
                            )

                            logger.info("Operational optimisation started !")
                            # print(f"Optimisation of : {override}")
                            run_hourly_optimisation(
                                override=override,
                                optim_mode="operational",
                                static_overrides=static_override_dict,
                                is_nested=is_nested,
                                pool=rl_env,
                            )
                        else:
                            logger.info("Design optimisation is not performed yet!")
                    except KeyboardInterrupt:
                        logger.warning("User interrupted the process.")

                    finally:
                        # catch-all cleanup
                        if rl_env is not None:
                            logger.info("Terminating RL environments...")
                            rl_env.close()
                            logger.info("RL environments closed.")
                else:
                    logger.info(f"{opt_type} is not a valid optimiser!")

        else:
            logger.info(f"{CONFIG['route']} : Not a valid route!")
