__author__ = "Aryan Chaudhary"

__all__ = [
    "run_deap_ga_optimisation",
    "run_fmincon_optimisation",
    "run_ga_optimisation",
    "run_nlopt",
    "run_pyga_optimisation",
    "run_scipy_minimise",
    "train_rl",
]

from .ga_optimiser.deap_ga_optimiser import run_deap_ga_optimisation

# from .deap_ga_parallel import GAEngine
from .fmincon_optimiser import run_fmincon_optimisation
from .matlab_ga_optimiser import run_ga_optimisation
from .nlopt_fmincon import run_nlopt
from .pygad_ga_optimiser import run_pyga_optimisation
from .scipy_fmincon import run_scipy_minimise
from .rl_optimiser.ppo_rl_training import train_rl
