import os


def calculate_batch_size(population_size, mpi_size, num_days):
    """
    Calculate individuals per batch using available CPU resources.

    Assumes one MPI rank per node and a local multiprocessing pool
    on each rank.
    """
    if population_size < 1:
        return 1

    if mpi_size < 1 or num_days < 1:
        raise ValueError("mpi_size and num_days must be >= 1")

    workers_per_rank = int(os.environ.get("LOCAL_WORKERS", "112"))

    if workers_per_rank < 1:
        raise ValueError("LOCAL_WORKERS must be >= 1")

    # Each individual needs one simulation per representative day.
    individuals_per_rank = max(1, workers_per_rank // num_days)

    # Never create batches larger than the population.
    return min(population_size, individuals_per_rank)
