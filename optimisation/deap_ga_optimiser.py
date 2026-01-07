import random
import numpy as np
import mlflow
import json
import hashlib
from multiprocessing import Pool, cpu_count
from pathlib import Path
import pickle
from functools import partial

from deap import base, creator, tools, algorithms
from prefect import task
from prefect.logging import get_run_logger

from utilities.checkpointing import atomic_pickle_dump
from config import CONFIG
from simulation import run_simulation
from objective_functions import objective_function


# ----------------------------
# GLOBALS for multiprocessing
# ----------------------------
_GA_VAR_NAMES = None
_GA_VAR_TYPES = None
_GA_LB = None
_GA_UB = None
_GA_STATIC_OVERRIDES = None


def init_worker(
    var_names,
    var_types,
    lb,
    ub,
    static_overrides,
):
    global _GA_VAR_NAMES, _GA_VAR_TYPES, _GA_LB, _GA_UB, _GA_STATIC_OVERRIDES

    _GA_VAR_NAMES = var_names
    _GA_VAR_TYPES = var_types
    _GA_LB = lb
    _GA_UB = ub
    _GA_STATIC_OVERRIDES = static_overrides


# ============================================================
# FITNESS FUNCTION (MUST BE TOP-LEVEL)
# ============================================================
def deap_fitness(individual, hour):
    overrides_dyn = {
        _GA_VAR_NAMES[i]: _GA_VAR_TYPES[i](individual[i])
        for i in range(len(_GA_VAR_NAMES))
    }

    final_overrides = {**overrides_dyn, **_GA_STATIC_OVERRIDES}

    sim_result = run_simulation(final_overrides)

    obj = objective_function(
        sim_result["hourly_energy"],
        sim_result["pc_htf_pump_power"],
        sim_result["field_htf_pump_power"],
        hour_index=hour,
    )
    fitness = float(obj)
    return (fitness,)


# ============================================================
# POPULATION EVALUATION
# ============================================================
def evaluate_population(toolbox, population):
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    return len(invalid_ind)


# ============================================================
# MAIN GA TASK
# ============================================================
@task()
def run_deap_ga_optimisation(
    override: dict,
    static_overrides: dict[str, float],
    is_nested: bool,
    curr_hour: int,
):
    logger = get_run_logger()

    if mlflow.active_run() and not is_nested:
        mlflow.end_run()

    with mlflow.start_run(run_name=f"GA_hour_{curr_hour}", nested=is_nested):
        mlflow.set_tag("Author", CONFIG["author"])
        mlflow.log_artifact("config.py")

        # ----------------------------
        # Read configuration
        # ----------------------------
        var_names = override["overrides"]
        var_types = override["types"]
        lb, ub = override["lb"], override["ub"]

        pop_size = CONFIG["sol_per_pop"]
        num_generations = CONFIG["num_generations"]
        cxpb = CONFIG.get("cxpb", 0.5)
        mutpb = CONFIG.get("mutpb", 0.2)
        indpb = CONFIG.get("indpb", 0.2)

        logger.info(f"Variables: {var_names}")
        logger.info(f"Types: {var_types}")
        logger.info(f"Bounds: {list(zip(lb, ub))}")

        # ----------------------------
        # DEAP setup
        # ----------------------------
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        def gen_individual():
            return [random.uniform(lb[i], ub[i]) for i in range(len(var_names))]

        toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            gen_individual,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", partial(deap_fitness, hour=curr_hour))
        toolbox.register(
            "select", tools.selTournament, tournsize=CONFIG.get("tournament_size", 3)
        )
        toolbox.register("mate", tools.cxOnePoint)

        def custom_mutation(individual):
            for i in range(len(individual)):
                if random.random() < indpb:
                    if var_types[i] is int:
                        individual[i] = random.randint(lb[i], ub[i])
                    else:
                        sigma = 0.1 * (ub[i] - lb[i])
                        individual[i] += random.gauss(0, sigma)

                    individual[i] = max(lb[i], min(ub[i], individual[i]))
            return (individual,)

        toolbox.register("mutate", custom_mutation)

        # ----------------------------
        # Multiprocessing pool (spawn-safe)
        # ----------------------------
        n_cores = min(cpu_count(), CONFIG.get("num_cores", cpu_count()))
        logger.info(f"{n_cores} cores working!")

        pool = Pool(
            processes=n_cores,
            initializer=init_worker,
            initargs=(
                var_names,
                var_types,
                lb,
                ub,
                static_overrides,
            ),
        )

        toolbox.register("map", pool.map)
        toolbox.register("evaluate", partial(deap_fitness, hour=curr_hour))

        # ----------------------------
        # Stable checkpoint key
        # ----------------------------
        ckpt_key = hashlib.sha256(
            json.dumps(
                {
                    "vars": var_names,
                    "lb": lb,
                    "ub": ub,
                    "pop": pop_size,
                    "gens": num_generations,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]

        checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints" / ckpt_key
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        resume_file = checkpoint_dir / "checkpoint_latest.pkl"

        # ----------------------------
        # Resume or fresh start
        # ----------------------------
        if resume_file.exists() and CONFIG.get("resume_from_checkpoint", False):
            logger.info(f"Resuming from {resume_file}")
            with open(resume_file, "rb") as f:
                cp = pickle.load(f)

            pop = cp["pop"]
            logbook = cp["logbook"]
            hof = cp["hof"]
            start_gen = cp["generation"] + 1
            random.setstate(cp["rndstate"])
            np.random.set_state(cp["np_rndstate"])
        else:
            logger.info("Starting fresh GA run")
            random.seed(CONFIG.get("random_seed"))
            np.random.seed(CONFIG.get("random_seed"))

            pop = toolbox.population(n=pop_size)
            logbook = tools.Logbook()
            logbook.header = ["gen", "nevals", "avg", "max"]
            hof = tools.HallOfFame(maxsize=5)
            start_gen = 0

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        if start_gen == 0:
            evaluate_population(toolbox, pop)
            hof.update(pop)

        # ----------------------------
        # GA loop
        # ----------------------------
        for gen in range(start_gen, num_generations):
            offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
            nevals = evaluate_population(toolbox, offspring)

            pop = toolbox.select(offspring, k=pop_size)
            hof.update(pop)

            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=nevals, **record)

            mlflow.log_metrics(
                {"gen_avg": record["avg"], "gen_max": record["max"]},
                step=gen,
            )

            cp_data = {
                "pop": pop,
                "logbook": logbook,
                "hof": hof,
                "generation": gen,
                "rndstate": random.getstate(),
                "np_rndstate": np.random.get_state(),
            }

            atomic_pickle_dump(cp_data, resume_file)

            if gen % CONFIG.get("checkpoint_interval", 2) == 0:
                gen_file = checkpoint_dir / f"checkpoint_gen_{gen}.pkl"
                atomic_pickle_dump(cp_data, gen_file)
                mlflow.log_artifact(str(gen_file), artifact_path="checkpoints/history")

        # ----------------------------
        # Cleanup pool
        # ----------------------------
        pool.close()
        pool.join()

        # ----------------------------
        # Final result
        # ----------------------------
        best_ind = hof[0]
        best_solution = [
            var_types[i](
                max(
                    lb[i],
                    min(
                        ub[i],
                        round(best_ind[i]) if var_types[i] is int else best_ind[i],
                    ),
                )
            )
            for i in range(len(var_names))
        ]

        best_fitness = float(best_ind.fitness.values[0])

        mlflow.log_metrics({"Best fitness": best_fitness})
        mlflow.log_params(
            {
                "pop_size": pop_size,
                "num_generations": num_generations,
                "cxpb": cxpb,
                "mutpb": mutpb,
                "checkpoint_key": ckpt_key,
            }
        )

        # ----------------------------
        # Pretty console output (same style as your original code)
        # ----------------------------
        if CONFIG.get("verbose", True):
            print("\n" + "-" * 40)
            print(f"Results for hour = {curr_hour}")
            print("\n" + "-" * 40)
            print("Final Best Solutions")
            print("-" * 40)

            for i, name in enumerate(var_names):
                val = best_solution[i]

                # Match formatting you used earlier
                if isinstance(val, float):
                    print(f"  {name:20}: {val:.4f}")
                else:
                    print(f"  {name:20}: {val}")

            print("-" * 40)
            print(f"{'Best Fitness':20}: {best_fitness:.6f}")
            print("-" * 40)

        logger.info(f"Best solution: {best_solution}")
        logger.info(f"Best fitness: {best_fitness}")

        return (
            best_solution,
            best_fitness,
            {
                "pop": pop,
                "logbook": logbook,
                "hof": hof,
            },
        )
