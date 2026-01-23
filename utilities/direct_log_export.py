import sys
import os
import logging
from prefect.context import get_run_context


class TeedLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Instant disk write

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_file_logging():
    try:
        ctx = get_run_context()
        flow_run_id = str(ctx.flow_run.id)
        flow_run_name = ctx.flow_run.name
    except Exception:
        flow_run_id = "unknown"
        flow_run_name = "manual_run"

    os.makedirs("flow_logs", exist_ok=True)
    log_filename = f"flow_logs/{flow_run_name}_{flow_run_id[:8]}.log"

    # 1. Hijack the system output for prints
    logger_instance = TeedLogger(log_filename)
    sys.stdout = logger_instance
    sys.stderr = logger_instance

    # 2. Force Prefect to use our new hijacked stream
    # We target the root 'prefect' logger
    prefect_logger = logging.getLogger("prefect")

    # Remove existing handlers that are tied to the old terminal
    for handler in prefect_logger.handlers[:]:
        prefect_logger.removeHandler(handler)

    # Add our new handler pointing to the hijacked sys.stdout
    new_handler = logging.StreamHandler(sys.stdout)
    new_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    )
    prefect_logger.addHandler(new_handler)

    # Crucial: Ensure the logger is actually enabled
    prefect_logger.setLevel(logging.INFO)

    # Do the same for flow_runs and task_runs specifically
    logging.getLogger("prefect.flow_runs").addHandler(new_handler)
    logging.getLogger("prefect.task_runs").addHandler(new_handler)

    print(f"--- LOG SESSION STARTED: {log_filename} ---")
    return log_filename
