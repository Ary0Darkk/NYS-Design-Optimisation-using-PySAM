import logging
import sys
import os
import multiprocessing
from datetime import datetime

# These are the actual codes the terminal understands
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGNETA = "\033[95m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"
LIGHT_GRAY = "\033[2m"


def setup_custom_logger(
    name="NYS_Optimisation",
    log_folder="logs",
    existing_file=None,
):
    """Log to the console and a timestamped, rank-specific file."""

    # Race-safe directory creation
    os.makedirs(log_folder, exist_ok=True)

    # Identify the MPI rank; defaults to 0 outside MPI.
    rank = os.environ.get("OMPI_COMM_WORLD_RANK", "0")

    # Use the supplied file or generate a rank-specific filename.
    if existing_file:
        log_file = existing_file
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{name}_{timestamp}_rank{rank}.log"
        log_file = os.path.join(log_folder, filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid adding duplicate handlers in the same process.
    if logger.handlers:
        return logger

    file_formatter = logging.Formatter(
        "%(asctime)s | PID:%(process)-5d | %(levelname)-8s | "
        "%(filename)s:%(funcName)s:%(lineno)d | %(message)s"
    )

    console_formatter = logging.Formatter(
        f"{BLUE}%(asctime)s{RESET} | "
        f"{MAGNETA}PID:%(process)-5d{RESET} | "
        f"{GREEN}%(levelname)-8s{RESET} | "
        "%(filename)s:%(funcName)s:%(lineno)d | "
        f"{LIGHT_GRAY}%(message)s{RESET}"
    )

    file_handler = logging.FileHandler(
        log_file,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    if multiprocessing.current_process().name == "MainProcess":
        logger.info(
            "Logger initialized | MPI rank: %s | File: %s",
            rank,
            log_file,
        )

    return logger


# This allows you to import 'logger' directly in other files
# logger = logging.getLogger("NYS_Optimisation")
