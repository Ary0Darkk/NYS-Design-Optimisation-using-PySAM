# file to route design optimisation and operational one
from typing import Optional

from optimisation.ga_optimiser import run_ga_optimisation
from optimisation.fmincon_optimiser import run_fmincon_optimisation
from optimisation.pygad_ga_optimiser import run_pyga_optimisation
from optimisation.scipy_fmincon import run_scipy_minimise
from optimisation.nlopt_fmincon import run_nlopt
from optimisation.deap_ga_optimiser import run_deap_ga_optimisation

from config import CONFIG


def optimisation_mode() -> str:
    override = None
    if CONFIG["route"] == "design":
        override = "deisgn_overrides"
    elif CONFIG["route"] == "operational":
        override = "operational_overrides"
    elif CONFIG["route"] == "des-operational":
        override = "deisgn_operational"
    else:
        print(f"{CONFIG['route']} : Not found! ")

    return override


def call_optimiser(
    override: list[float], static_overrides: Optional[dict[str, float]] = None
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
                override=override, static_overrides=static_overrides
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


def run_router():
    # design and operational optim logic
    optimals = None

    override = optimisation_mode()
    print(f"Optimisation mode : {override}")

    # only perfoms single optim
    if override != "deisgn_operational":
        call_optimiser(override)
    # perform both optim in sequence
    else:
        optimals = call_optimiser(override="deisgn_overrides")

        if optimals is not None:
            design_dict = dict(zip(CONFIG["deisgn_overrides"], optimals[0]))

            call_optimiser(
                override="operational_overrides", static_overrides=design_dict
            )

        else:
            print("Design optimisation is not performed yet!")
