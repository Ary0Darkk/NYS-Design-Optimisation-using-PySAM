import random
import numpy as np
import pandas as pd
import mlflow
import json
import hashlib
import logging
from pathlib import Path
import pickle
import tabulate as tb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from deap import base, creator, tools, algorithms

from config import CONFIG
from simulation import run_simulation
from objective_functions import objective_function


logger = logging.getLogger("NYS_Optimisation")

matplotlib.use("Agg")  # headless, SSH-safe


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


# fitness function
def deap_fitness(individual, hour, optim_mode, var_names, var_types, static_overrides):
    # Use the passed-in variables instead of Globals
    overrides_dyn = {
        var_names[i]: var_types[i](individual[i]) for i in range(len(var_names))
    }

    final_overrides = {**overrides_dyn, **static_overrides}

    sim_result, penality_flag = run_simulation(final_overrides)

    if optim_mode == "design":
        try:
            if penality_flag is not True:
                obj = sim_result["annual_energy"]
            else:
                obj = CONFIG["penalty"]
            return (obj,)
        except KeyError:
            # This will print the ACTUAL keys being returned by the cached task
            print(
                f"CRITICAL: 'annual_energy' missing. Available keys: {list(sim_result.keys())}"
            )
            return (0.0,)  # Return a penalty score instead of crashing
    elif optim_mode == "operational":
        if penality_flag is not True:
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
            obj = CONFIG["penalty"]
    else:
        print(f"{optim_mode} is invalid!")

    fitness = np.float32(obj)
    return (fitness,)


# population evaluation
def evaluate_population(toolbox, population):
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    return len(invalid_ind)


def init_fresh_ga(toolbox, pop_size):
    """Encapsulates the logic for starting a brand-new evolution."""
    logger.info("Starting fresh GA run")

    # Set seeds from CONFIG for reproducibility
    random.seed(CONFIG.get("random_seed"))
    np.random.seed(CONFIG.get("random_seed"))

    pop = toolbox.population(n=pop_size)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "std", "max", "min"]

    hof = tools.HallOfFame(maxsize=CONFIG.get("hall_of_fame_size"))

    start_gen = 0
    return pop, logbook, hof, start_gen


# serialise population object to be pickle-safe
def serialize_population(pop):
    return [(list(ind), ind.fitness.values) for ind in pop]


# de-serialise population object from pickle one
def deserialize_population(serialized, toolbox):
    pop = []
    for genome, fitness in serialized:
        ind = toolbox.individual()
        ind[:] = genome
        ind.fitness.values = fitness
        pop.append(ind)
    return pop


# -------- MAIN GA TASK ----------------------
def run_deap_ga_optimisation(
    override: dict,
    optim_mode: str,
    static_overrides: dict[str, float],
    is_nested: bool,
    curr_hour: int,
    pool,
    rec,
):
    try:
        timestamp = CONFIG["session_time"]
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
            if optim_mode == "operational":
                mlflow.log_param("year", 2020)
                mlflow.log_param("operating_start_hour", 7)
                mlflow.log_param("operating_end_hour", 16)
                mlflow.set_tag(f"hour_{curr_hour}_season", rec["season"])
                mlflow.set_tag(
                    f"hour_{curr_hour}_date",
                    f"{rec['day']:02d}-{rec['month']:02d}-2020",
                )
                mlflow.set_tag(f"hour_{curr_hour}_hod", rec["hour_of_day"])

            # ----------------------------
            # Read configuration
            # ----------------------------
            var_names = override["overrides"]
            var_types = override["types"]
            lb, ub = override["lb"], override["ub"]

            pop_size = CONFIG.get("pop_size")
            num_generations = CONFIG.get("num_generations")
            cxpb = CONFIG.get("cxpb")
            mutpb = CONFIG.get("mutpb")
            indpb = CONFIG.get("indpb")

            logger.info(f"Variables: {var_names}")
            logger.info(f"Types: {var_types}")
            logger.info(f"Bounds: {list(zip(lb, ub))}")

            # DEAP setup
            if not hasattr(creator, "FitnessMax"):
                creator.create(
                    "FitnessMax", base.Fitness, weights=(1.0,)
                )  # 1.0 for maximising objective func
            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.FitnessMax)

            # toolbox init
            toolbox = base.Toolbox()

            # individual generation
            def gen_individual():
                individual = []
                for i in range(len(var_names)):
                    if var_types[i] is int:
                        val = random.randint(lb[i], ub[i])
                    else:
                        # Scale by 100 to get 2 decimal places
                        # e.g., for range 0 to 10, pick a random int between 0 and 1000, then / 100
                        scaled_lb = int(lb[i] * 100)
                        scaled_ub = int(ub[i] * 100)
                        val = random.randrange(scaled_lb, scaled_ub + 1) / 100.0
                    individual.append(val)
                return individual

            toolbox.register(
                "individual",
                tools.initIterate,
                creator.Individual,
                gen_individual,
            )
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register(
                "evaluate",
                deap_fitness,
                optim_mode=optim_mode,
                hour=curr_hour,
                var_names=var_names,
                var_types=var_types,
                static_overrides=static_overrides,
            )
            toolbox.register(
                "select",
                tools.selTournament,
                tournsize=CONFIG.get("tournament_size"),
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

                        # clip to boundaries
                        individual[i] = max(lb[i], min(ub[i], individual[i]))

                        # handle types and precision
                        if var_types[i] is int:
                            individual[i] = int(round(individual[i]))
                        else:
                            # Round to 2 decimal places for floats
                            individual[i] = round(float(individual[i]), 2)

                return (individual,)

            toolbox.register("mutate", custom_mutation)
            # pool mapping
            toolbox.register("map", pool.map)

            # ------- Stable checkpoint key ------------------
            ckpt_key = hashlib.sha256(
                json.dumps(
                    {
                        "optim_mode": optim_mode,
                        "vars": var_names,
                        "types": [t.__name__ for t in var_types],
                        "lb": lb,
                        "ub": ub,
                        "pop": pop_size,
                        "gens": num_generations,
                        "cxpb": cxpb,
                        "mutpb": mutpb,
                        "indpb": indpb,
                        "static_overrides": static_overrides,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:12]

            def safe_pickle_save(data, file_path):
                tmp_path = file_path.with_suffix(".tmp")
                try:
                    with open(tmp_path, "wb") as f:
                        pickle.dump(data, f)
                    tmp_path.replace(file_path)  # atomic on same filesystem
                    return True
                except Exception as e:
                    logger.warning(f"Checkpoint save failed: {e}")
                    return False

            # checkpoint_dir = (
            #     Path(__file__).resolve().parents[1] / "checkpoints" / ckpt_key
            # )

            # BASE_DIR = Path(__file__).resolve().parents[1] # Directory of the current script
            sub_path = "GA_design" if optim_mode == "design" else "GA_operational"
            checkpoint_dir = Path(f"checkpoints/GA/{sub_path}/{ckpt_key}")

            resume_file = Path(f"{checkpoint_dir}/checkpoint_latest.pkl")
            resume_file.parent.mkdir(parents=True, exist_ok=True)

            # ------- Resume or fresh start -----------------------
            if resume_file.exists() and CONFIG.get("resume_from_checkpoint", False):
                try:
                    with open(resume_file, "rb") as f:
                        cp = pickle.load(f)

                    # VALIDATION: Ensure checkpoint matches current config
                    if cp.get("ckpt_key") != ckpt_key:
                        logger.warning(
                            "Checkpoint key mismatch! Starting fresh to avoid DNA corruption."
                        )
                        pop, logbook, hof, start_gen = init_fresh_ga(toolbox, pop_size)
                    else:
                        logger.info(
                            f"Resuming from {resume_file} at generation {cp['generation']}"
                        )
                        # init random seed first
                        random.setstate(cp["rndstate"])
                        np.random.set_state(cp["np_rndstate"])
                        pop = deserialize_population(cp["population"], toolbox)
                        logbook = cp["logbook"]
                        hof = tools.HallOfFame(maxsize=CONFIG.get("hall_of_fame_size"))
                        hof[:] = deserialize_population(cp["hof"], toolbox)
                        start_gen = cp["generation"] + 1
                except Exception as e:
                    logger.error(f"Checkpoint corrupted: {e}. Starting fresh.")
                    pop, logbook, hof, start_gen = init_fresh_ga(toolbox, pop_size)
            else:
                pop, logbook, hof, start_gen = init_fresh_ga(
                    toolbox, pop_size
                )  # Standard Start

            stats = tools.Statistics(
                lambda ind: ind.fitness.values[0]
            )  # take 0th-index because deap supports multi-objective function
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            if start_gen == 0:
                evaluate_population(toolbox, pop)
                hof.update(pop)

            gens_log = []
            max_fitness_log = []
            avg_fitness_log = []

            if optim_mode == "design":
                plot_path = Path(
                    f"plots/GA_plots/GA_design_fitness_vs_gen_{timestamp}.png"
                )
            else:
                plot_path = Path(
                    f"plots/GA_plots/GA_operational_fitness_vs_gen_{timestamp}.png"
                )
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            # ----------- GA loop ------------------------------------
            for gen in range(start_gen, num_generations):
                offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
                nevals = evaluate_population(toolbox, offspring)

                pop = toolbox.select(offspring, k=pop_size)
                hof.update(pop)

                record = stats.compile(pop)
                logbook.record(gen=gen, nevals=nevals, **record)

                gens_log.append(gen)
                max_fitness_log.append(record["max"])
                avg_fitness_log.append(record["avg"])

                if gen % 1 == 0 or gen == num_generations - 1:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    ax.plot(gens_log, max_fitness_log, "r-", label="Max Fitness", lw=2)
                    ax.plot(
                        gens_log, avg_fitness_log, "b--", label="Avg Fitness", alpha=0.7
                    )

                    ax.set_title("Fitness vs Generation")
                    ax.set_xlabel("Generation")
                    ax.set_ylabel("Fitness")
                    ax.xaxis.set_major_locator(MultipleLocator(1))
                    ax.legend()
                    ax.grid(True, linestyle=":", alpha=0.5)

                    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
                    plt.close(fig)

                mlflow.log_metrics(
                    {"gen_avg": record["avg"], "gen_max": record["max"]},
                    step=gen,
                )
                best_ind = hof[0]
                mlflow.log_metrics(
                    {
                        name: v_type(
                            max(l, min(u, round(val) if v_type is int else val))
                        )
                        for name, v_type, l, u, val in zip(
                            var_names, var_types, lb, ub, best_ind
                        )
                    },
                    step=gen,
                )
                mlflow.log_metrics(
                    {"best fitness": float(best_ind.fitness.values[0])},
                    step=gen,
                )

                cp_data = {
                    "var_name": var_names,
                    "var_types": var_types,
                    "lb": lb,
                    "ub": ub,
                    "pop": serialize_population(pop),
                    "logbook": logbook,
                    "hof": serialize_population(hof),
                    "ckpt_key": ckpt_key,
                    "generation": gen,
                    "rndstate": random.getstate(),
                    "np_rndstate": np.random.get_state(),
                }

                try:
                    # saves "latest" checkpoint
                    safe_pickle_save(cp_data, resume_file)

                    # Periodic history checkpoint
                    if gen % CONFIG.get("checkpoint_interval") == 0:
                        gen_file = Path(f"{checkpoint_dir}/checkpoint_gen_{gen}.pkl")
                        gen_file.parent.mkdir(parents=True, exist_ok=True)
                        safe_pickle_save(cp_data, gen_file)
                        mlflow.log_artifact(
                            str(gen_file), artifact_path="checkpoints/history"
                        )
                except Exception as e:
                    logger.warning(f"Checkpoint write failed (Generation {gen}): {e}")

                # def save_to_json(cp_data,file_name):
                #     with open(file_name, "w") as f:
                #         json.dump(cp_data, f, indent=4)

                # # Save the "latest" checkpoint directly
                # save_to_json(cp_data, resume_file)

                # # Periodic history checkpoint
                # if gen % CONFIG.get("checkpoint_interval", 5) == 0:
                #     gen_file = Path(f"{checkpoint_dir}/ checkpoint_gen_{gen}.json")
                #     gen_file.parent.mkdir(parents=True, exist_ok=True)
                #     save_to_json(cp_data, gen_file)

                #     # Ensure file is written before logging to MLflow
                #     if gen_file.exists():
                #         mlflow.log_artifact(str(gen_file), artifact_path="checkpoints/history")

            # save as a static image at the end
            # fig.savefig(fname=file_name)
            # plt.show() # blocks execution of code

            # ------ Final result ---------------------------------
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

            mlflow.log_params(
                {
                    "pop_size": pop_size,
                    "num_generations": num_generations,
                    "indpb": indpb,
                    "cxpb": cxpb,
                    "mutpb": mutpb,
                    "checkpoint_key": ckpt_key,
                }
            )

            # ------ console output -------------------
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

            # mlflow.log_metrics(res_dict)
            result_logbook = pd.DataFrame([res_dict])
            result_logbook.index = result_logbook.index + 1
            result_logbook.index.name = "serial"

            if optim_mode == "design":
                file_name = Path(f"results/GA_results/GA_design_{timestamp}.csv")
            else:
                file_name = Path(f"results/GA_results/GA_operational_{timestamp}.csv")
            file_name.parent.mkdir(parents=True, exist_ok=True)

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
        print("Closed GA optimisation!")
