# file to route design optimisation and operational one
from optimisation.ga_optimiser import run_ga_optimisation
from optimisation.fmincon_optimiser import run_fmincon_optimisation
from optimisation.pygad_ga_optimiser import run_pyga_optimisation
from optimisation.scipy_fmincon import run_scipy_minimise
from optimisation.nlopt_fmincon import run_nlopt
from optimisation.deap_ga_optimiser import run_deap_ga_optimisation

from simulation.simulation import run_simulation
from config import CONFIG


def optimisation_mode()->str:
    override = None
    if CONFIG["route"] == "design":
       override = "deisgn_overrides"
    elif CONFIG["route"] == "operational":
        override = "operational_overrides"
    elif CONFIG["route"] == "des-operational":
        override = "deisgn_operational"
    else:
        print(f'{CONFIG["route"]} : Not found! ')
    
    return override


def call_optimiser(
        override:list[float],
        static_overrides:dict[str,float]
):
    # FIXME : try and catch is not working as expected,
    # look at keyboard interupt working
    try:
        # optimisation
        if CONFIG["optimiser"] == "fmincon":
            x_opt, f_val = run_fmincon_optimisation()
        elif CONFIG["optimiser"] == "ga":
            x_opt, f_val = run_ga_optimisation()
        elif CONFIG["optimiser"] == "pygad_ga":
            x_opt, f_val, _ = run_pyga_optimisation()
        elif CONFIG["optimiser"] == "nlopt":
            x_opt, f_val, _ = run_nlopt()
        elif CONFIG["optimiser"] == "scipy_min":
            x_opt, f_val, _ = run_scipy_minimise()
        elif CONFIG["optimiser"] == "scipy_min":
            x_opt, f_val, _ = run_scipy_minimise()
        elif CONFIG["optimiser"] == "deap_ga":
            x_opt, f_val, _ = run_deap_ga_optimisation(override=override,static_overrides=static_overrides)
        else:
            print(f'{CONFIG["optimiser"]} : Not an optimiser')
        # disp optimal values
        print(f"x_opt : {x_opt} \nf_val : {f_val}")
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user. Stopping...\n")
    except Exception as e:
        print("Unexpected error :", e)

    return x_opt,f_val


def main():

    # design and operational optim logic
    optimals = None

    override = optimisation_mode()
    print(f'Optimisation mode : {override}')

    # only perfoms single optim
    if override != "deisgn_operational":
        call_optimiser(override)
    # perform both optim in sequence
    else:
        optimals = call_optimiser(override="deisgn_overrides")

        if optimals is not None:
            design_dict = dict(zip(CONFIG["deisgn_overrides"],optimals[0]))

            call_optimiser(override="operational_overrides",
                           static_overrides=design_dict)
            
        else:
            print('Design optimisation is not performed yet!')


if __name__ == "__main__":
    main()