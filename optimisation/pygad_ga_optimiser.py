import pygad
import config
from simulation.simulation import run_simulation

def run_pyga_optimisation(num_generations=200, sol_per_pop=40, num_parents_mating=10,
           mutation_num_genes=1, random_seed=None, verbose=True):
    """
    Runs GA to maximize annual energy.
    Returns: best_solution (list), best_fitness (float), ga_instance
    """

    cfg = config.CONFIG
    var_names = cfg["overrides"]
    lb = cfg["lb"]
    ub = cfg["ub"]

    # ---- fitness function (MUST have 3 params) ----
    def fitness_func(ga_instance, solution, solution_idx):
        overrides = {var_names[i]: float(solution[i]) for i in range(len(var_names))}
        annual_energy = run_simulation(overrides)
        return float(annual_energy)  # maximize

    # ---- gene space from bounds ----
    gene_space = [{"low": lb[i], "high": ub[i]} for i in range(len(var_names))]

    ga = pygad.GA(
        num_generations=num_generations,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents_mating,

        num_genes=len(var_names),
        gene_space=gene_space,

        fitness_func=fitness_func,

        parent_selection_type="tournament",
        crossover_type="single_point",

        mutation_type="random",
        mutation_num_genes=mutation_num_genes,  # good for 2 vars

        random_seed=random_seed,
        suppress_warnings=verbose
    )

    ga.run()

    best_solution, best_fitness, _ = ga.best_solution()

    if verbose:
        print("\nBest solution (overrides):")
        for i, name in enumerate(var_names):
            print(f"  {name}: {best_solution[i]:.6f}")
        print("Max annual energy:", best_fitness)

    return best_solution, best_fitness, ga
