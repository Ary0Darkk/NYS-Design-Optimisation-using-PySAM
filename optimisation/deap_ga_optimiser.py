import random
import numpy as np
import mlflow
import pickle
import sys
import json
from pathlib import Path
from deap import base, creator, tools, algorithms

from prefect import task
from prefect.logging import get_run_logger

# Setup path to internal modules
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from config import CONFIG
from simulation.simulation import run_simulation
from objective_functions.objective_func import objective_function


@task()
def run_deap_ga_optimisation(
    override, static_overrides: dict[str, float], is_nested: bool
):
    logger = get_run_logger()

    # ---- MLFLOW SETUP ----
    if mlflow.active_run() and not is_nested:
        mlflow.end_run()

    run_name = CONFIG["run_name"]

    with mlflow.start_run(run_name=run_name, nested=is_nested):
        mlflow.set_tag("Author", CONFIG["author"])

        # logs config file
        mlflow.log_artifact("config.py")

        var_names = override["overrides"]
        logger.info(f"Override variables : {var_names}")
        var_types = override["types"]
        logger.info(f"Data-Type of variables : {var_types}")
        lb, ub = override["lb"], override["ub"]
        logger.info(f"Lower bound : {lb}")
        logger.info(f"Upper bound : {ub}")
        pop_size = CONFIG["sol_per_pop"]
        logger.info(f"Population size : {pop_size}")
        num_generations = CONFIG["num_generations"]
        logger.info(f"Num of gen : {num_generations}")
        cxpb = CONFIG.get("cxpb", 0.5)
        logger.info(f"cxpb value : {cxpb}")
        mutpb = CONFIG.get("mutpb", 0.2)
        logger.info(f"mutpb value : {mutpb}")

        # ---- SETUP DEAP GLOBALS ----
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        # --- TOOLBOX CONFIGURATION ----
        toolbox = base.Toolbox()

        # generates individual
        def gen_individual():
            return [random.uniform(lb[i], ub[i]) for i in range(len(var_names))]

        toolbox.register(
            "individual", tools.initIterate, creator.Individual, gen_individual
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def fitness_func_individual(individual):
            t_overrides = {}
            for i in range(len(var_names)):
                val = individual[i]

                t_overrides[var_names[i]] = var_types[i](val)

            # Combine with static overrides
            final_overrides = {**t_overrides, **static_overrides}

            # Run the simulation
            sim_result = run_simulation(final_overrides)
            obj = objective_function(
                sim_result["hourly_energy"],
                sim_result["pc_htf_pump_power"],
                sim_result["field_htf_pump_power"],
            )

            return (float(obj[0]),)

        toolbox.register("evaluate", fitness_func_individual)
        toolbox.register(
            "select", tools.selTournament, tournsize=CONFIG.get("tournament_size")
        )
        toolbox.register("mate", tools.cxOnePoint)

        def custom_mutation(individual, indpb, var_types, low, up):
            for i in range(len(individual)):
                if random.random() < indpb:
                    if var_types[i] == int:
                        # Integer Mutation: Uniform choice within bounds
                        individual[i] = random.randint(low[i], up[i])
                    else:
                        # Float Mutation: Gaussian shift
                        # Sigma is 10% of the range
                        sigma = (up[i] - low[i]) * 0.1
                        individual[i] += random.gauss(0, sigma)

                    # Boundary Enforcement (Crucial for MATLAB-style consistency)
                    individual[i] = max(low[i], min(up[i], individual[i]))

            return (individual,)

        # Register using 'partial' logic (passing the types and bounds once)
        toolbox.register(
            "mutate",
            custom_mutation,
            indpb=CONFIG.get("indpb"),
            var_types=var_types,
            low=lb,
            up=ub,
        )

        # ---- CHECKPOINT / RESUME LOGIC ----
        run_id = mlflow.active_run().info.run_id
        checkpoint_dir = Path.home() / "My Drive" / "ga_checkpoints" / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # We always resume from 'checkpoint_latest.pkl' if it exists
        resume_file = checkpoint_dir / "checkpoint_latest.pkl"

        if resume_file.exists() and CONFIG.get("resume_from_checkpoint", False):
            print(f"--- Resuming from {resume_file} ---")
            with open(resume_file, "rb") as f:
                cp = pickle.load(f)

            pop = cp["pop"]
            logbook = cp["logbook"]
            hof = cp["hof"]
            start_gen = cp["generation"] + 1
            random.setstate(cp["rndstate"])
            np.random.set_state(cp["np_rndstate"])
        else:
            print("--- Starting fresh optimization ---")
            if CONFIG.get("random_seed") is not None:
                random.seed(CONFIG["random_seed"])
                np.random.seed(CONFIG["random_seed"])
            pop = toolbox.population(n=pop_size)
            logbook = tools.Logbook()
            logbook.header = ["gen", "nevals"] + ["avg", "max"]
            hof = tools.HallOfFame(maxsize=5)  # Keeps top 5 individuals ever found
            start_gen = 0

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        # ---- EVOLUTIONARY LOOP ----
        if start_gen == 0:
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            hof.update(pop)

        for gen in range(start_gen, num_generations):
            offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.select(offspring, k=pop_size)
            hof.update(pop)

            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            mlflow.log_metrics(
                {"gen_avg": record["avg"], "gen_max": record["max"]}, step=gen
            )

            # SAVE NAMED CHECKPOINTS
            cp_data = {
                "pop": pop,
                "logbook": logbook,
                "hof": hof,
                "generation": gen,
                "rndstate": random.getstate(),
                "np_rndstate": np.random.get_state(),
            }

            # ---- Save as latest for auto-resume ----
            with open(resume_file, "wb") as f:
                pickle.dump(cp_data, f)

            # ---- Save generation-specific for retrieval (e.g., every 10 gens) ----
            if gen % CONFIG.get("checkpoint_interval", 2) == 0:
                gen_file = checkpoint_dir / f"checkpoint_gen_{gen}.pkl"
                with open(gen_file, "wb") as f:
                    pickle.dump(cp_data, f)
                mlflow.log_artifact(str(gen_file), artifact_path="checkpoints/history")

        # 6. FINAL RESULTS (Using Hall of Fame)
        best_ind = hof[0]  # The best ever found

        # We transform the raw GA "genotype" (floats) into your "phenotype" (ints/floats)
        best_solution = []
        for i in range(len(var_names)):
            val = best_ind[i]

            # Apply rounding and clamping if the type is int
            if var_types[i] is int:
                val = round(val)
                val = max(lb[i], min(ub[i], val))

            # Cast to the final type (float or int) defined in your config
            best_solution.append(var_types[i](val))

        best_fitness = float(best_ind.fitness.values[0])

        # Use the "reported" version for MLflow and dictionary logging
        x_dict = {
            f"{name} optimal value": val for name, val in zip(var_names, best_solution)
        }

        # Log the clean, rounded/typed results
        mlflow.log_metrics({**x_dict, "Best fitness": best_fitness})

        # Print the cleaned results for your console
        if CONFIG.get("verbose", True):
            print("\n" + "-" * 35)
            print("Final Best Solutions:")
            print("-" * 35)
            for i, name in enumerate(var_names):
                print(f"  {name:20}: {best_solution[i]:.4f}")
            print("-" * 35)
            print(f"{'Best Fitness':15}: {best_fitness:.4f}")
            print("-" * 35)

        # Capture all relevant GA and Problem settings
        params_to_log = {
            "pop_size": pop_size,
            "num_generations": num_generations,
            "cxpb": cxpb,
            "mutpb": mutpb,
            "tournament_size": CONFIG.get("tournament_size", 3),
            "mutation_num_genes": CONFIG.get("mutation_num_genes", 1),
            "random_seed": str(CONFIG["random_seed"]),
            "variable_names": str(var_names),
            "lower_bounds": str(lb),
            "upper_bounds": str(ub),
            "resume_enabled": CONFIG.get("resume_from_checkpoint", False),
        }

        params_dict = json.dumps(params_to_log, indent=4)
        logger.info(f"Optimization Results:\n{params_dict}")

        # Log everything to MLflow
        mlflow.log_params(params_to_log)

        return best_solution, best_fitness, {"pop": pop, "logbook": logbook, "hof": hof}
