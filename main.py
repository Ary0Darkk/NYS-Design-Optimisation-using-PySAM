# from simulation.simulation import run_simulation
from optimisation.optimiser import run_optimisation


def main():
    
    # simulation
    # sim_output = run_simulation()
    # print(f'Annual Energy : {sim_output["annual_energy"]}')
    
    # optimisation
    x_opt, f_val = run_optimisation()
    print(f'x_opt : {x_opt} \nf_val : {f_val}')
    
    
if __name__ == "__main__":
    main()