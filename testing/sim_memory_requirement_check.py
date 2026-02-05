import psutil
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.simulation import _run_simulation_core


# get pid of simualtion script
def run_mem_check_on_simulation(overrides):
    pid = os.getpid()
    print(f"PID of this process: {pid=}")
    process = psutil.Process(pid)

    _run_simulation_core(overrides)

    # Memory in MB
    mem_info = process.memory_info().rss / (1024 * 1024)
    print(f"Task {overrides} used {mem_info:.2f} MB")
    # return results


design_overrides = {
    "specified_total_aperture": 8175,
    "Row_Distance": 4,
    "ColperSCA": 12,
    "W_aperture": 10,
    "L_SCA": 120,
}

operational_overrides = {
    "m_dot": 4,
    "T_startup": 320,
    "T_shutdown": 320,
}

all_overrides = {**design_overrides, **operational_overrides}


if __name__ == "__main__":
    # try with all three possible cases to know that one is not marginally larger than another
    run_mem_check_on_simulation(all_overrides)
