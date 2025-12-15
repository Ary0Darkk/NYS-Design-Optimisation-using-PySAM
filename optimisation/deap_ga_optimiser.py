import random
import numpy as np
import mlflow
import dagshub
from deap import base, creator, tools, algorithms
from config import CONFIG
from simulation.simulation import run_simulation

def run_deap_ga_optimisation():
    """
    Runs GA (DEAP) to maximize annual energy.
    Returns: best_solution (list), best_fitness (float), ga_instance (dict)
    """
    # database setup
    mlflow.set_tracking_uri("https://dagshub.com/aryanvj787/NYS-Design-Optimisation-using-PySAM.mlflow")
    dagshub.init(repo_owner='aryanvj787', repo_name='NYS-Design-Optimisation-using-PySAM', mlflow=True)

    # set experiment name
    mlflow.set_experiment("Deap-ga-optimisation")

    # SAFETY: close any run that may be active from earlier imports/calls
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # set run name here
    run_name = CONFIG["run_name"]

    with mlflow.start_run(run_name=run_name):

        # ---- author tag ----
        mlflow.set_tag("Author", CONFIG["author"])

        var_names = CONFIG["overrides"]
        print(type(var_names))
        print(len(var_names))
        lb = CONFIG["lb"]
        ub = CONFIG["ub"]

        mlflow.log_params({
            "Lower Bound": lb,
            "Upper Bound": ub,
        })

        # ---- DEAP setup ----
        random_seed = CONFIG.get("random_seed", None)
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # create Maximizing fitness and Individual
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # per-gene attribute generator (captures differing lb/ub per gene)
        def gen_individual():
            return [random.uniform(lb[i], ub[i]) for i in range(len(var_names))]

        toolbox.register("individual", tools.initIterate, creator.Individual, gen_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # ---- fitness function ----
        def fitness_func_individual(individual):
            # DEAP passes only the individual to the fitness function here
            overrides = {var_names[i]: float(individual[i]) for i in range(len(var_names))}
            sim_result = run_simulation(overrides)
            return (float(sim_result["monthly_energy"][0]),)  # DEAP expects a tuple

        toolbox.register("evaluate", fitness_func_individual)

        # selection, crossover, mutation
        toolbox.register("select", tools.selTournament, tournsize=CONFIG.get("tournament_size", 3))
        toolbox.register("mate", tools.cxOnePoint)

        # mutation: replace `mutation_num_genes` randomly-chosen genes with uniform samples within bounds
        def mut_random_replace(individual, mutation_num_genes):
            size = len(individual)
            # pick unique gene indices to mutate
            idxs = random.sample(range(size), min(mutation_num_genes, size))
            for idx in idxs:
                individual[idx] = random.uniform(lb[idx], ub[idx])
            # enforce bounds (clamp) just in case
            for i in range(size):
                if individual[i] < lb[i]:
                    individual[i] = lb[i]
                elif individual[i] > ub[i]:
                    individual[i] = ub[i]
            return individual,

        toolbox.register("mutate", mut_random_replace, mutation_num_genes=CONFIG.get("mutation_num_genes", 1))

        # GA configuration - map from CONFIG where possible
        pop_size = CONFIG["sol_per_pop"]
        num_generations = CONFIG["num_generations"]
        cxpb = CONFIG.get("cxpb", 0.5)   # crossover probability per pair
        mutpb = CONFIG.get("mutpb", 0.2) # probability to apply mutation to an offspring

        config_vars = {
            "num_generations": num_generations,
            "pop_size": pop_size,
            "tournament_size": CONFIG.get("tournament_size", 3),
            "num_genes": len(var_names),
            "gene_space": [{"low": lb[i], "high": ub[i]} for i in range(len(var_names))],
            "cxpb": cxpb,
            "mutpb": mutpb,
            "mutation_num_genes": CONFIG.get("mutation_num_genes", 1),
            "random_seed": random_seed,
            "suppress_warnings": CONFIG.get("verbose", False)
        }

        # log config_vars to mlflow (flattened)
        mlflow.log_params({k: (v if not isinstance(v, list) else str(v)) for k, v in config_vars.items()})

        # initialize population
        pop = toolbox.population(n=pop_size)

        # optional stats to observe progress
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # run the evolutionary algorithm
        # we use eaSimple which applies selection, crossover (cxpb), mutation (mutpb) per generation
        pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                           ngen=num_generations, stats=stats, verbose=CONFIG.get("verbose", False))

        # find best individual in final population
        best_ind = tools.selBest(pop, 1)[0]
        best_solution = list(map(float, best_ind))
        best_fitness = float(best_ind.fitness.values[0])

        # find index of best in final population (closest equivalent to pygad's best_match_idx)
        try:
            best_match_idx = next(i for i, ind in enumerate(pop) if ind is best_ind)
        except StopIteration:
            best_match_idx = -1

        x_dict = {}
        for var, value in zip(CONFIG["overrides"], best_solution):
            rav = var + " optimal value"
            x_dict[rav] = float(value)

        mlflow.log_metrics(x_dict)
        mlflow.log_metrics({
            "Best fitness": best_fitness,
            "Best matching index": best_match_idx,
        })

        if CONFIG.get("verbose", True):
            print("\nBest solution:")
            for i, name in enumerate(var_names):
                print(f"  {name}: {best_solution[i]:.6f}")
            print("Max annual energy:", best_fitness)

        # Build a ga-like instance to return for introspection
        ga_instance = {
            
            "toolbox": toolbox,
            "population": pop,
            "logbook": logbook,
            "stats": stats
        }

        return best_solution, best_fitness, ga_instance
