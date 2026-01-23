import random
import numpy as np
import pandas as pd
import mlflow
import json
import hashlib
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
import pickle
from functools import partial
import tabulate as tb
from joblib import hash as joblib_hash

from deap import base, creator, tools, algorithms

from utilities.checkpointing import atomic_pickle_dump
from config import CONFIG
from simulation import run_simulation
from objective_functions import objective_function

logger = logging.getLogger("NYS_Optimisation")


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
def deap_fitness(individual, hour, optim_mode: str):
    overrides_dyn = {
        _GA_VAR_NAMES[i]: _GA_VAR_TYPES[i](individual[i])
        for i in range(len(_GA_VAR_NAMES))
    }

    final_overrides = {**overrides_dyn, **_GA_STATIC_OVERRIDES}

    sim_result = run_simulation(final_overrides)
    # table = tb.tabulate(
    #     [final_overrides.values()], headers=final_overrides.keys(), tablefmt="psql"
    # )
    # logger.info(f"Ran sim with paramters :\n{table}")

    if optim_mode == "design":
        try:
            obj = sim_result["annual_energy"]
            return (obj,)
        except KeyError:
            # This will print the ACTUAL keys being returned by the cached task
            print(
                f"CRITICAL: 'annual_energy' missing. Available keys: {list(sim_result.keys())}"
            )
            return (0.0,)  # Return a penalty score instead of crashing
    elif optim_mode == "operational":
        obj = objective_function(
            sim_result["hourly_energy"],
            sim_result["pc_htf_pump_power"],
            sim_result["field_htf_pump_power"],
            sim_result["field_collector_tracking_power"],
            sim_result["pc_startup_thermal_power"],
            sim_result["field_piping_thermal_loss"],
            sim_result["receiver_thermal_loss"],
            hour_index=hour,
        )
    else:
        print(f"{optim_mode} is invalid!")

    fitness = np.float32(obj)
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


def run_deap_ga_optimisation(
    override: dict,
    optim_mode: str,
    static_overrides: dict[str, float],
    is_nested: bool,
    curr_hour: int,
):
    try:
        if optim_mode == "design":
            run_name = "GA_Design_optimisation"
        else:
            run_name = f"GA_hour_{curr_hour}"

        if mlflow.active_run() and not is_nested:
            mlflow.end_run()

        with mlflow.start_run(run_name=run_name, nested=is_nested):
            mlflow.set_tag("Author", CONFIG["author"])
            mlflow.log_artifact("config.py")
            mlflow.set_tag("hour", curr_hour)

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

            toolbox.register(
                "evaluate", partial(deap_fitness, optim_mode=optim_mode, hour=curr_hour)
            )
            toolbox.register(
                "select",
                tools.selTournament,
                tournsize=CONFIG.get("tournament_size", 3),
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
            toolbox.register(
                "evaluate", partial(deap_fitness, optim_mode=optim_mode, hour=curr_hour)
            )

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

            checkpoint_dir = (
                Path(__file__).resolve().parents[1] / "checkpoints" / ckpt_key
            )
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
                    mlflow.log_artifact(
                        str(gen_file), artifact_path="checkpoints/history"
                    )

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
            res_dict = {}
            header_line = "-" * 40
            if optim_mode == "design":
                for i, name in enumerate(var_names):
                    val = best_solution[i]
                    res_dict[name] = val  # stores in dict to save data in csv

                res_dict["best_fitness"] = best_fitness
                #  formats output
                res_table = tb.tabulate(res_dict.items(), tablefmt="grid")
                logger.info(
                    f"\n{header_line}\n"
                    f"GA DESIGN OPTIMAL SOLUTIONS\n"
                    f"{header_line}\n"
                    f"Final Best Results\n"
                    f"{res_table}"
                )
            else:
                res_dict["hour"] = curr_hour
                for i, name in enumerate(var_names):
                    val = best_solution[i]
                    res_dict[name] = val  # stores in dict to save data in csv

                res_dict["best_fitness"] = best_fitness

                # formats output
                res_table = tb.tabulate(res_dict.items(), tablefmt="grid")
                logger.info(
                    f"\n{header_line}\n"
                    f"GA Optimal solution (hour {curr_hour})\n"
                    f"{header_line}\n"
                    f"Final Best Results\n"
                    f"{res_table}"
                )

            result_logbook = pd.DataFrame([res_dict])
            result_logbook.index = result_logbook.index + 1
            result_logbook.index.name = "serial"

            if optim_mode == "design":
                file_name = Path("results/GA_design_results.csv")
            else:
                file_name = Path("results/GA_operational_results.csv")
            file_name.parent.mkdir(exist_ok=True)

            file_exists = file_name.exists()
            result_logbook.to_csv(file_name, mode="a", header=not file_exists)

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

    except KeyboardInterrupt:
        print("Interrupted by User!\nStopping...")

    finally:
        # clean pool
        pool.close()
        pool.join()
        print("Flushed all processes!")
