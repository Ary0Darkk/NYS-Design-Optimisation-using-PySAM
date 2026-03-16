import random
import numpy as np
from deap import base, creator, tools, algorithms

from simulation.simulation import run_simulation
from demand_data import get_dynamic_price


class SolarGAConfig:
    """Genetic Algorithm specific parameters."""

    POP_SIZE = 5
    GENS = 2
    CXPB = 0.1  # Crossover probability
    MUTPB = 0.2  # Mutation probability
    INDPB = 0.05  # Probability of mutating an individual's specific gene
    TOURN_SIZE = 2


class SolarGeneticOptimizer:
    def __init__(self, var_config: dict):
        """
        var_config: dict with 'names', 'types', 'lb', 'ub' for GA to evolve
        static_overrides: fixed parameters for every simulation
        """
        self.var_names = var_config["names"]
        self.var_types = var_config["types"]
        self.lb = var_config["lb"]
        self.ub = var_config["ub"]

        # Load pricing data once
        self.price_df = get_dynamic_price()
        self.price_values = self.price_df["dynamic_price"].values.flatten()

        # Constants from previous logic
        self.USD_TO_INR = 73
        self.MWT_TO_MWE = 0.4
        self.LAND_EXPECTANCY = 30
        self.KW_CONV = 1000

        self._setup_deap()

    def _setup_deap(self):
        """Initializes the DEAP toolbox and types."""
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Individual Creation
        def create_ind():
            ind = []
            for i in range(len(self.var_names)):
                if self.var_types[i] is int:
                    ind.append(random.randint(self.lb[i], self.ub[i]))
                else:
                    ind.append(random.uniform(self.lb[i], self.ub[i]))
            return ind

        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, create_ind
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "select", tools.selTournament, tournsize=SolarGAConfig.TOURN_SIZE
        )

        # Custom mutation to handle bounds and types
        def mutate(individual):
            for i in range(len(individual)):
                if random.random() < SolarGAConfig.INDPB:
                    if self.var_types[i] is int:
                        individual[i] = random.randint(self.lb[i], self.ub[i])
                    else:
                        # Gaussian mutation for floats
                        individual[i] += random.gauss(
                            0, (self.ub[i] - self.lb[i]) * 0.1
                        )
                        individual[i] = max(self.lb[i], min(self.ub[i], individual[i]))
            return (individual,)

        self.toolbox.register("mutate", mutate)

    def evaluate(self, individual):
        """The core objective function adapted for GA."""
        # Map individual to simulation overrides
        current_overrides = {
            self.var_names[i]: self.var_types[i](individual[i])
            for i in range(len(self.var_names))
        }
        final_params = {**current_overrides}

        # Run Sim
        sim_result, penalty_flag = run_simulation(final_params)

        if penalty_flag:
            # NEW: Print the failing parameters to your terminal
            print(f"DEBUG: Penalty Triggered for {current_overrides}")
            return (-999999999.0,)  # Massive penalty for invalid designs

        # Calculate Objective (Previous Prompt Logic)
        try:
            energy_term = (
                sim_result["hourly_energy"] * self.KW_CONV
                - (
                    sim_result["field_htf_pump_power"]
                    + sim_result["pc_htf_pump_power"]
                    + sim_result["field_collector_tracking_power"]
                )
                * self.KW_CONV
                - (
                    sim_result["pc_startup_thermal_power"]
                    + sim_result["field_piping_thermal_loss"]
                    + sim_result["receiver_thermal_loss"]
                )
                * self.MWT_TO_MWE
                * self.KW_CONV
            )

            total_revenue = sum(energy_term * self.price_values)
            land_cost = (
                float(sim_result["land_cost"]) * self.USD_TO_INR
            ) / self.LAND_EXPECTANCY

            return (total_revenue - land_cost,)
        except Exception:
            return (-999999999.0,)

    def evolve(self):
        """Runs the actual Genetic Algorithm loop."""
        pop = self.toolbox.population(n=SolarGAConfig.POP_SIZE)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        print(f"Starting evolution for variables: {self.var_names}")

        pop, log = algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=SolarGAConfig.CXPB,
            mutpb=SolarGAConfig.MUTPB,
            ngen=SolarGAConfig.GENS,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

        best_ind = hof[0]
        best_fitness = best_ind.fitness.values[0]

        # Convert best individual to readable dict
        best_params = {
            self.var_names[i]: self.var_types[i](best_ind[i])
            for i in range(len(self.var_names))
        }

        return best_params, best_fitness


# --- EXECUTION ---
if __name__ == "__main__":
    # Define what the GA should evolve
    variables_to_optimize = {
        "names": ["specified_total_aperture", "Row_Distance", "ColperSCA"],
        "types": [int, float, int],
        "lb": [5000, 5.0, 4],
        "ub": [12000, 15.0, 10],
    }

    optimizer = SolarGeneticOptimizer(variables_to_optimize)
    best_config, max_profit = optimizer.evolve()

    print("\n" + "=" * 30)
    print("GA OPTIMIZATION COMPLETE")
    print("=" * 30)
    print(f"Best Parameters: {best_config}")
    print(f"Max Annual Profit (INR): {max_profit:,.2f}")
