import pygad
import mlflow
import dagshub
from config import CONFIG
from simulation.simulation import run_simulation


def run_pyga_optimisation():
    """
    Runs GA to maximize annual energy.
    Returns: best_solution (list), best_fitness (float), ga_instance
    """
    # database setup
    mlflow.set_tracking_uri(
        "https://dagshub.com/aryanvj787/NYS-Design-Optimisation-using-PySAM.mlflow"
    )
    dagshub.init(
        repo_owner="aryanvj787",
        repo_name="NYS-Design-Optimisation-using-PySAM",
        mlflow=True,
    )

    # set experiment name
    mlflow.set_experiment("Pygad-ga-optimisation")

    # SAFETY: close any run that may be active from earlier imports/calls
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # set run name here
    run_name = CONFIG["run_name"]

    with mlflow.start_run(run_name=run_name):
        # author tag
        mlflow.set_tag("Author", CONFIG["author"])

        var_names = CONFIG["overrides"]
        print(type(var_names))
        print(len(var_names))
        lb = CONFIG["lb"]
        ub = CONFIG["ub"]

        mlflow.log_params(
            {
                "Lower Bound": lb,
                "Upper Bound": ub,
            }
        )

        # ---- fitness function (MUST have 3 params) ----
        def fitness_func(ga_instance, solution, solution_idx):
            overrides = {
                var_names[i]: float(solution[i]) for i in range(len(var_names))
            }
            annual_energy = run_simulation(overrides)
            return float(annual_energy)  # maximize

        # ---- gene space from bounds ----
        gene_space = [{"low": lb[i], "high": ub[i]} for i in range(len(var_names))]
        config_vars = {
            "num_generations": CONFIG["num_generations"],
            "sol_per_pop": CONFIG["sol_per_pop"],
            "num_parents_mating": CONFIG["num_parents_mating"],
            "num_genes": len(var_names),
            "gene_space": gene_space,
            "fitness_func": fitness_func,
            "parent_selection_type": "tournament",
            "crossover_type": "single_point",
            "mutation_type": "random",
            "mutation_num_genes": CONFIG["mutation_num_genes"],  # good for 2 vars
            "random_seed": CONFIG["random_seed"],
            "suppress_warnings": CONFIG["verbose"],
        }

        mlflow.log_params(config_vars)
        # ga = pygad.GA(
        #     num_generations=num_generations,
        #     sol_per_pop=sol_per_pop,
        #     num_parents_mating=num_parents_mating,

        #     num_genes=len(var_names),
        #     gene_space=gene_space,

        #     fitness_func=fitness_func,

        #     parent_selection_type="tournament",
        #     crossover_type="single_point",

        #     mutation_type="random",
        #     mutation_num_genes=mutation_num_genes,  # good for 2 vars

        #     random_seed=random_seed,
        #     suppress_warnings=verbose
        # )

        ga = pygad.GA(**config_vars)

        ga.run()

        best_solution, best_fitness, best_match_idx = ga.best_solution()

        x_dict = {}
        for var, value in zip(CONFIG["overrides"], best_solution):
            rav = var + " optimal value"
            x_dict[rav] = float(value)

        mlflow.log_metrics(x_dict)
        mlflow.log_metrics(
            {
                "Best fitness": best_fitness,
                "Best matching index": best_match_idx,
            }
        )

        if CONFIG["verbose"]:
            print("\nBest solution:")
            for i, name in enumerate(var_names):
                print(f"  {name}: {best_solution[i]:.6f}")
            print("Max annual energy:", best_fitness)

        return best_solution, best_fitness, ga
