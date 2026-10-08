import argparse
import multiprocessing as mp
import json

import mlflow
from mpi4py import MPI

from utilities.setup_custom_logger import setup_custom_logger

from .optimisation.ga_optimiser.deap_ga_optimiser import (
    run_deap_ga_optimisation,
    worker_loop,
)

from config import CONFIG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Season-wise CSP genetic algorithm optimisation"
    )

    parser.add_argument(
        "--season",
        required=True,
        choices=["winter", "summer", "monsoon", "post"],
        help="Seasonal optimisation case to run",
    )

    return parser.parse_args()


def log_experiment_parameters(season, mpi_size):
    """
    Log all fixed experiment configuration once at the beginning.
    """

    design = CONFIG["design"]
    operational = CONFIG["operational"]

    mlflow.log_params(
        {
            # Experiment
            "season": season,
            "random_seed": CONFIG.get("random_seed"),
            "penalty": CONFIG.get("penalty"),
            # GA
            "population_size": CONFIG.get("pop_size"),
            "num_generations": CONFIG.get("num_generations"),
            "cxpb": CONFIG.get("cxpb"),
            "mutpb": CONFIG.get("mutpb"),
            "indpb": CONFIG.get("indpb"),
            "tournament_size": CONFIG.get("tournament_size"),
            "hall_of_fame_size": CONFIG.get("hall_of_fame_size"),
            # Problem
            "num_days": len(CONFIG["SEASONS"][season]),
            # HPC
            "mpi_ranks": mpi_size,
            "cores_per_rank": 112,
            "total_cores": mpi_size * 112,
            # Variables
            "design_variables": json.dumps(design["overrides"]),
            "operational_variables": json.dumps(operational["overrides"]),
            # Bounds
            "design_lower_bounds": json.dumps(design["lb"]),
            "design_upper_bounds": json.dumps(design["ub"]),
            "operational_lower_bounds": json.dumps(operational["lb"]),
            "operational_upper_bounds": json.dumps(operational["ub"]),
        }
    )


def main():
    logger = setup_custom_logger()
    logger.info("NYS-Optimisation started!")

    args = parse_args()
    season = args.season

    # ---------------------------------------------------------
    # MPI
    # ---------------------------------------------------------

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger.info(f"[Rank {rank}] Starting optimisation for season: {season}")

    # =========================================================
    # MASTER
    # =========================================================

    if rank == 0:
        # -----------------------------------------------------
        # MLflow configuration
        # -----------------------------------------------------

        mlflow.set_experiment("CSP_Seasonal_GA")

        with mlflow.start_run(run_name=f"GA_{season}"):
            # Tags = metadata describing the run
            mlflow.set_tags(
                {
                    "project": "CSP Plant Optimisation",
                    "season": season,
                    "scheduler": "PBS",
                    "parallelization": "MPI + multiprocessing",
                    "algorithm": "DEAP Genetic Algorithm",
                    "simulation": "PySAM",
                }
            )

            # Parameters = fixed configuration
            log_experiment_parameters(
                season=season,
                mpi_size=size,
            )

            logger.info(
                f"[Rank 0] MLflow run started: {mlflow.active_run().info.run_id}"
            )

            # -------------------------------------------------
            # Local multiprocessing pool
            # -------------------------------------------------

            local_pool = mp.Pool(processes=112)

            try:
                run_deap_ga_optimisation(
                    season=season,
                    local_pool=local_pool,
                )

            finally:
                # Tell remote MPI ranks to stop
                for worker in range(1, size):
                    comm.send(None, dest=worker, tag=1)

                local_pool.close()
                local_pool.join()

            logger.info("[Rank 0] MLflow run completed")

    # =========================================================
    # WORKERS
    # =========================================================

    else:
        worker_loop(season=season)

    logger.info("Optimisation Completed!")


if __name__ == "__main__":
    main()
