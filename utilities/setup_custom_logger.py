import logging
import sys
import os
from datetime import datetime

# These are the actual codes the terminal understands
BLUE   = "\033[94m"
CYAN   = "\033[96m"
MAGNETA= "\033[95m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
LIGHT_GRAY="\033[2m"


def setup_custom_logger(name="NYS_Optimisation", log_folder="logs",existing_file=None):
    """Sets up a logger that outputs to both console and a timestamped file."""

    # Create logs directory if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # If a file is passed (from a worker), use it. Otherwise, create a new one.
    if existing_file:
        log_file = existing_file
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M") # Removed seconds for stability
        log_file = os.path.join(log_folder, f"{name}_{timestamp}.log")

    # create a master logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # If handlers already exist, don't log "initialized" again, just return the logger
    if logger.handlers:
        return logger

    # create Formatters (Time | Level | Message)
    # file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    # Detailed Formatter (Highly recommended for complex simulations)
    file_formatter = logging.Formatter(
        "%(asctime)s | PID:%(process)-5d | %(levelname)-8s | %(filename)s:%(funcName)s:%(lineno)d | %(message)s"
    )
    console_formatter = logging.Formatter(
        f"{BLUE}%(asctime)s{RESET} | {MAGNETA}PID:%(process)-5d{RESET} | {GREEN}%(levelname)-8s{RESET} | {CYAN}%(filename)s:%(funcName)s:%(lineno)d{RESET} | {LIGHT_GRAY}%(message)s{RESET}"
    )
    # console_formatter = logging.Formatter('%(levelname)-8s | %(message)s')

    # File Handler (Instant Flush)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(file_formatter)

    # console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Writing to: {log_file}")
    return logger


# This allows you to import 'logger' directly in other files
# logger = logging.getLogger("NYS_Optimisation")
