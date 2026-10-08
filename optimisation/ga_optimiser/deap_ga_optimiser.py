import random
import json
import hashlib
import logging
import pickle
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tabulate as tb

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from deap import base, creator, tools, algorithms
from mpi4py import MPI

from config import CONFIG
from ...simulation.simulation import run_simulation
from ...objective_functions.objective_func import objective_function
from ...utilities.index_creation import get_season_days

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger = logging.getLogger("NYS_Optimisation")


def worker_loop(season):
    """
    Worker MPI rank.

    Each worker rank owns one compute node and creates
    a local multiprocessing pool.
    """

    print(f"[Rank {rank}] Starting worker for season: {season}")

    pool = mp.Pool(processes=112)

    try:
        while True:
            batch = comm.recv(
                source=0,
                tag=1,
            )

            # None means shutdown
            if batch is None:
                break

            print(f"[Rank {rank}] Received batch of {len(batch)} individuals")

            fitnesses = evaluate_batch(
                batch=batch,
                pool=pool,
                season=season,
            )

            comm.send(
                fitnesses,
                dest=0,
                tag=2,
            )

    finally:
        pool.close()
        pool.join()

        print(f"[Rank {rank}] Worker stopped")


def batch_individuals(individuals, batch_size=16):
    """
    Split individuals into batches
    """
    return [
        individuals[i : i + batch_size] for i in range(0, len(individuals), batch_size)
    ]


def evaluate_batch(batch, pool, season):
    """
    Evaluate a batch of individuals over all representative
    days belonging to the selected season.
    """

    season_days = get_season_days(season)

    tasks = []

    for individual_id, individual in enumerate(batch):
        for day_info in season_days:
            tasks.append(
                (
                    individual_id,
                    day_info["local_index"],
                    day_info["month"],
                    day_info["day"],
                    day_info["day_index"],
                    individual,
                )
            )

    logger.info(
        f"[Rank {rank}] "
        f"Created {len(tasks)} simulation tasks "
        f"for {len(batch)} individuals"
    )

    results = pool.starmap(
        run_one_simulation,
        tasks,
    )

    # One total fitness per individual
    fitnesses = [0.0] * len(batch)

    for (
        individual_id,
        local_day_index,
        month,
        day,
        day_index,
        objective,
    ) in results:
        fitnesses[individual_id] += objective

    return [(fitness,) for fitness in fitnesses]


def distribute_batches(
    batches,
    local_pool,
    season,
):
    """
    Distribute batches across MPI worker ranks.

    Rank 0 evaluates one batch locally while remote MPI
    ranks evaluate their assigned batches.
    """

    workers = list(range(1, size))

    all_fitnesses = []

    for wave_start in range(
        0,
        len(batches),
        size,
    ):
        wave = batches[wave_start : wave_start + size]

        logger.info(
            f"[Rank 0] Processing batches {wave_start + 1} - {wave_start + len(wave)}"
        )

        # -------------------------------------------------
        # Send batches to remote MPI ranks
        # -------------------------------------------------

        remote_batches = wave[: len(workers)]

        for worker, batch in zip(
            workers,
            remote_batches,
        ):
            comm.send(
                batch,
                dest=worker,
                tag=1,
            )

        # -------------------------------------------------
        # Rank 0 evaluates local batch
        # -------------------------------------------------

        local_result = None

        if len(wave) > len(workers):
            local_batch = wave[len(workers)]

            logger.info("[Rank 0] Evaluating local batch")

            local_result = evaluate_batch(
                batch=local_batch,
                pool=local_pool,
                season=season,
            )

        # -------------------------------------------------
        # Receive remote results
        # -------------------------------------------------

        remote_results = []

        for worker in workers[: len(remote_batches)]:
            result = comm.recv(
                source=worker,
                tag=2,
            )

            remote_results.extend(result)

        # -------------------------------------------------
        # Preserve batch order
        # -------------------------------------------------

        all_fitnesses.extend(remote_results)

        if local_result is not None:
            all_fitnesses.extend(local_result)

    return all_fitnesses


def evaluate_population(
    population,
    local_pool,
    season,
):
    """
    Evaluate an entire DEAP population.
    """

    batches = batch_individuals(
        population,
        batch_size=16,
    )

    fitnesses = distribute_batches(
        batches=batches,
        local_pool=local_pool,
        season=season,
    )

    for individual, fitness in zip(
        population,
        fitnesses,
    ):
        individual.fitness.values = fitness

    return len(population)


def run_one_simulation(
    individual_id,
    local_day_index,
    month,
    day,
    day_index,
    individual,
):
    # -------------------------------------------------
    # Design parameters
    # -------------------------------------------------

    design_params = individual[:5]

    # -------------------------------------------------
    # Operational parameters
    # -------------------------------------------------

    operational_params = individual[5:]

    # local_day_index = 0, 1, ..., 6
    #
    # This determines which 3 operational genes belong
    # to this representative day.

    start = local_day_index * 3
    end = start + 3

    daily_operational = operational_params[start:end]

    # -------------------------------------------------
    # Combine inputs
    # -------------------------------------------------

    simulation_input = design_params + daily_operational

    # -------------------------------------------------
    # Convert to PySAM overrides
    # -------------------------------------------------

    var_names = CONFIG["design"]["overrides"] + CONFIG["operational"]["overrides"]

    var_types = CONFIG["design"]["types"] + CONFIG["operational"]["types"]

    overrides = {
        var_names[j]: var_types[j](simulation_input[j]) for j in range(len(var_names))
    }

    # -------------------------------------------------
    # Run PySAM
    # -------------------------------------------------

    sim_result, penalty_flag = run_simulation(overrides)

    # -------------------------------------------------
    # Penalty
    # -------------------------------------------------

    if penalty_flag:
        return (
            individual_id,
            local_day_index,
            month,
            day,
            day_index,
            CONFIG["penalty"],
        )

    # -------------------------------------------------
    # Daily objective
    # -------------------------------------------------

    objective = objective_function(
        sim_result["hourly_energy"],
        sim_result["pc_htf_pump_power"],
        sim_result["field_htf_pump_power"],
        sim_result["field_collector_tracking_power"],
        sim_result["pc_startup_thermal_power"],
        sim_result["field_piping_thermal_loss"],
        sim_result["receiver_thermal_loss"],
        day_index=day_index,
    )

    # -------------------------------------------------
    # Validate objective
    # -------------------------------------------------

    if objective is None or not np.isfinite(objective):
        objective = CONFIG["penalty"]

    return (
        individual_id,
        local_day_index,
        month,
        day,
        day_index,
        float(objective),
    )


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
    season: str,
    local_pool,
):
    if season not in CONFIG["SEASONS"]:
        raise ValueError(
            f"Unknown season: {season}. "
            f"Available seasons: "
            f"{list(CONFIG['SEASONS'].keys())}"
        )

    season_days = get_season_days(season)

    num_days = len(season_days)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Starting GA optimisation for season: {season}")

    logger.info(f"Representative days: {season_days}")

    logger.info(f"Number of representative days: {num_days}")
    # Read configuration
    design_lb, design_ub = (
        CONFIG.get("design")["lb"],
        CONFIG.get("design")["ub"],
    )
    design_var = CONFIG.get("design")["overrides"]
    design_var_types = CONFIG.get("design")["types"]
    operational_lb, operational_ub = (
        CONFIG.get("operational")["lb"],
        CONFIG.get("operational")["ub"],
    )
    operational_var = CONFIG.get("operational")["overrides"]
    operational_var_types = CONFIG.get("operational")["types"]

    pop_size = CONFIG.get("pop_size")
    num_generations = CONFIG.get("num_generations")
    cxpb = CONFIG.get("cxpb")
    mutpb = CONFIG.get("mutpb")
    indpb = CONFIG.get("indpb")

    logger.info(f"Design Bounds: {list(zip(design_lb, design_ub))}")
    logger.info(f"Operational Bounds: {list(zip(operational_lb, operational_ub))}")

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
        # DESIGN VARIABLES
        for lb, ub in zip(
            design_lb,
            design_ub,
        ):
            individual.append(random.randint(lb, ub))

        # OPERATIONAL VARIABLES
        for _ in range(num_days):
            for lb, ub in zip(
                operational_lb,
                operational_ub,
            ):
                individual.append(random.randint(lb, ub))
        # Sanity check
        assert len(individual) == 5 + 3 * num_days
        return individual

    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        gen_individual,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # toolbox.register(
    #     "evaluate",
    #     deap_fitness,
    #     day_index=current_day,
    #     var_names=design_var + operational_var,
    #     var_types=design_var_types + operational_var_types,
    # )

    toolbox.register(
        "select",
        tools.selTournament,
        tournsize=CONFIG.get("tournament_size"),
    )
    toolbox.register("mate", tools.cxOnePoint)

    def custom_mutation(individual):
        """
        Mutate an individual consisting of:
            5 design parameters
            num_days × 3 operational parameters
        All variables are integer-valued.
        """
        # MUTATE DESIGN PARAMETERS
        for i in range(5):
            if random.random() < indpb:
                sigma = 0.1 * (design_ub[i] - design_lb[i])
                mutated_value = individual[i] + random.gauss(0, sigma)
                mutated_value = round(mutated_value)
                individual[i] = int(max(design_lb[i], min(design_ub[i], mutated_value)))
        # MUTATE OPERATIONAL PARAMETERS
        for day in range(num_days):
            # Starting index of this day's 3 variables
            start = 5 + day * 3
            for j in range(3):
                if random.random() < indpb:
                    sigma = 0.1 * (operational_ub[j] - operational_lb[j])
                    mutated_value = individual[start + j] + random.gauss(0, sigma)
                    mutated_value = round(mutated_value)
                    individual[start + j] = int(
                        max(
                            operational_lb[j],
                            min(operational_ub[j], mutated_value),
                        )
                    )

        return (individual,)

    toolbox.register("mutate", custom_mutation)

    # ------- Stable checkpoint key ------------------
    ckpt_key = hashlib.sha256(
        json.dumps(
            {
                "vars": design_var + operational_var,
                "types": [t.__name__ for t in design_var_types + operational_var_types],
                "lb": design_lb + operational_lb,
                "ub": design_ub + operational_ub,
                "pop": pop_size,
                "gens": num_generations,
                "cxpb": cxpb,
                "mutpb": mutpb,
                "indpb": indpb,
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

    checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints" / ckpt_key

    # BASE_DIR = (
    # Path(__file__).resolve().parents[1]
    # )  # Directory of the current script
    checkpoint_dir = Path(f"checkpoints/GA/{ckpt_key}")

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
        evaluate_population(
            population=pop,
            local_pool=local_pool,
            season=season,
        )
        hof.update(pop)

    gens_log = []
    max_fitness_log = []
    avg_fitness_log = []

    plot_path = Path(f"plots/GA_plots/{season}_fitness_vs_gen_{timestamp}.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    # ----------- GA loop ------------------------------------
    for gen in range(start_gen, num_generations):
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
        nevals = evaluate_population(
            population=offspring,
            local_pool=local_pool,
            season=season,
        )

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
            ax.plot(gens_log, avg_fitness_log, "b--", label="Avg Fitness", alpha=0.7)

            ax.set_title("Fitness vs Generation")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness")
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.5)

            fig.savefig(plot_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

        mlflow.log_metrics(
            {
                "fitness_avg": float(record["avg"]),
                "fitness_std": float(record["std"]),
                "fitness_min": float(record["min"]),
                "fitness_max": float(record["max"]),
                "nevals": float(nevals),
            },
            step=gen,
        )
        best_ind = hof[0]

        mlflow.log_metrics(
            {
                "best_fitness": float(best_ind.fitness.values[0]),
                "best_aperture": float(best_ind[0]),
                "best_row_distance": float(best_ind[1]),
                "best_col_per_sca": float(best_ind[2]),
                "best_w_aperture": float(best_ind[3]),
                "best_l_sca": float(best_ind[4]),
            },
            step=gen,
        )

        cp_data = {
            "var_name": design_var + operational_var,
            "var_types": design_var_types + operational_var_types,
            "lb": design_lb + operational_lb,
            "ub": design_ub + operational_ub,
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
                mlflow.log_artifact(str(gen_file), artifact_path="checkpoints/history")
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

    # =========================================================
    # FINAL RESULT
    # =========================================================

    best_ind = hof[0]

    best_fitness = float(best_ind.fitness.values[0])

    best_solution = {
        "season": season,
        "fitness": best_fitness,
        "individual": list(best_ind),
    }

    # ---------------------------------------------------------
    # Log final solution to MLflow
    # ---------------------------------------------------------

    mlflow.log_params(
        {
            "checkpoint_key": ckpt_key,
            "final_best_fitness": best_fitness,
        }
    )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    res_dict = {
        "season": season,
        "best_fitness": best_fitness,
    }

    for i, name in enumerate(design_var + operational_var):
        res_dict[name] = best_ind[i]

    res_table = tb.tabulate(
        res_dict.items(),
        tablefmt="grid",
    )

    logger.info(
        f"\n{'-' * 40}\n"
        f"GA Optimal solution ({season})\n"
        f"{'-' * 40}\n"
        f"Final Best Results\n"
        f"{res_table}"
    )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    result_logbook = pd.DataFrame([res_dict])

    result_logbook.index = result_logbook.index + 1

    result_logbook.index.name = "serial"

    file_name = Path(f"results/GA_results/{season}_{timestamp}.csv")

    file_name.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_logbook.to_csv(
        file_name,
        index=True,
    )

    # ---------------------------------------------------------
    # Save best solution JSON
    # ---------------------------------------------------------

    best_solution_path = Path(f"results/GA_results/best_solution_{season}.json")

    with open(
        best_solution_path,
        "w",
    ) as f:
        json.dump(
            best_solution,
            f,
            indent=4,
        )

    # ---------------------------------------------------------
    # MLflow artifacts
    # ---------------------------------------------------------

    mlflow.log_artifact(
        str(file_name),
        artifact_path="results",
    )

    mlflow.log_artifact(
        str(best_solution_path),
        artifact_path="results",
    )

    # Save GA logbook
    logbook_path = Path(f"results/GA_results/{season}_logbook.csv")

    pd.DataFrame(logbook).to_csv(
        logbook_path,
        index=False,
    )

    mlflow.log_artifact(
        str(logbook_path),
        artifact_path="ga_history",
    )

    # Log final plot
    if plot_path.exists():
        mlflow.log_artifact(
            str(plot_path),
            artifact_path="plots",
        )

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
