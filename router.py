# file to route design optimisation and operational one
from typing import Optional

from optimisation.ga_optimiser import run_ga_optimisation
from optimisation.fmincon_optimiser import run_fmincon_optimisation
from optimisation.pygad_ga_optimiser import run_pyga_optimisation
from optimisation.scipy_fmincon import run_scipy_minimise
from optimisation.nlopt_fmincon import run_nlopt
from optimisation.deap_ga_optimiser import run_deap_ga_optimisation

from config import CONFIG
import mlflow
import dagshub

from prefect import task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(
    cache_key_fn=task_input_hash, 
    persist_result=True,
    cache_expiration=timedelta(days=1),
    result_storage=CONFIG["storage_block"],
)
def optimisation_mode() -> str | dict[str, list[float]]:
    override = None
    if CONFIG["route"] == "design":
        override = CONFIG["design"]
    elif CONFIG["route"] == "operational":
        override = CONFIG["operational_overrides"]
    elif CONFIG["route"] == "des-operational":
        override = "design_operational"
    else:
        print(f"{CONFIG['route']} : Not found! ")

    return override

@task(
    cache_key_fn=task_input_hash, 
    persist_result=True,
    cache_expiration=timedelta(days=1),
    result_storage=CONFIG["storage_block"],
)
def call_optimiser(
    override: dict[str, list[float]],
    is_nested: bool,
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
            x_opt, f_val, _ = run_deap_ga_optimisation(
                override=override,
                static_overrides=static_overrides,
                is_nested=is_nested,
            )
        else:
            print(f"{opt_type} : Not a valid optimiser name in CONFIG")
        # only print if the variables were successfully set
        if x_opt is not None:
            print(f"x_opt : {x_opt} \nf_val : {f_val}")

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user. Stopping...\n")
    # except Exception as e:
    #     print("Unexpected error :", e)

    return x_opt, f_val

@task(
    cache_key_fn=task_input_hash, 
    persist_result=True,
    cache_expiration=timedelta(days=1),
    result_storage=CONFIG["storage_block"],
)
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

    # optimisation
    opt_type = CONFIG.get("optimiser")

    # set experiment name
    mlflow.set_experiment(f"{opt_type}-optimisation")

    # design and operational optim logic
    optimals = None

    override = optimisation_mode()

    # only perfoms single optim
    if override != "design_operational":
        print(f"{CONFIG['route']} optimisation started !")
        print(f"Optimisation of : {override}")
        call_optimiser(override, is_nested=False)
    # perform both optim in sequence
    else:
        is_nested = True  # informs mlflow for multi-step run
        # 1. Start a Parent Run to group everything
        with mlflow.start_run(run_name="Sequential des-operational"):
            print(
                "Going to begin Design plus Operational optimisation sequentially!\n\n"
            )
            print(f"Design optimisation started !")
            print(f"Optimisation of : {override}")
            optimals = call_optimiser(override=CONFIG["design"], is_nested=is_nested)

            if optimals is not None:
                design_dict = dict(zip(CONFIG["design"]["overrides"], optimals[0]))

                print(f"Operational optimisation started !")
                print(f"Optimisation of : {override}")
                call_optimiser(
                    override=CONFIG["operational"],
                    static_overrides=design_dict,
                    is_nested=is_nested,
                )

            else:
                print("Design optimisation is not performed yet!")
