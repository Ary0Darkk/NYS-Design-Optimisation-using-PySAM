# from simulation.simulation import run_simulation
from optimisation.ga_optimiser import run_ga_optimisation
from optimisation.fmincon_optimiser import run_fmincon_optimisation
from optimisation.pygad_ga_optimiser import run_pyga_optimisation
from optimisation.scipy_fmincon import run_scipy_minimise
from optimisation.nlopt_fmincon import run_nlopt
from config import CONFIG


def main():
    
    # simulation
    # sim_output = run_simulation()
    # print(f'Annual Energy : {sim_output["annual_energy"]}')
    
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
        # disp optimal values
        print(f'x_opt : {x_opt} \nf_val : {f_val}')
    except KeyboardInterrupt:
        print('\n\nOptimization interrupted by user. Stopping...\n')
    except Exception as e:
        print('Unexpected error :',e)
    
    
if __name__ == "__main__":
    main()