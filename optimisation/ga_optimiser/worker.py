import pickle
import argparse

from deap_ga_optimiser import deap_fitness

parser = argparse.ArgumentParser()
parser.add_argument("--gen", type=int)
parser.add_argument("--ind", type=int)

args = parser.parse_args()

with open(f"populations/gen_{args.gen}/ind_{args.ind}.pkl", "rb") as f:
    payload = pickle.load(f)

individual = payload["individual"]

fitness = deap_fitness(
    individual=individual,
    hour=payload["hour"],
    optim_mode=payload["optim_mode"],
    var_names=payload["var_names"],
    var_types=payload["var_types"],
    static_overrides=payload["static_overrides"],
)

with open(f"fitness_results/gen_{args.gen}/fitness_{args.ind}.pkl", "wb") as f:
    pickle.dump(fitness, f)
