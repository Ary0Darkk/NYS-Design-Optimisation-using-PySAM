import random
import numpy as np
import sys
from pathlib import Path
from deap import base, creator, tools, algorithms

import multiprocessing as mp

from prefect import task

# Setup path to internal modules
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)


@task()
class GAEngine:
    def __init__(
        self,
        var_names,
        var_types,
        lb,
        ub,
        objective_function,
        run_simulation,
        static_overrides,
        config,
        logger,
        mlflow,
    ):
        self.var_names = var_names
        self.var_types = var_types
        self.lb = lb
        self.ub = ub
        self.objective_function = objective_function
        self.run_simulation = run_simulation
        self.static_overrides = static_overrides
        self.CONFIG = config
        self.logger = logger
        self.mlflow = mlflow
        self.pool = mp.Pool(processes=self.CONFIG.get("n_jobs", mp.cpu_count()))
        self.toolbox.register("map", self.pool.map)

        self.toolbox = base.Toolbox()
        self._setup_toolbox()

    def _setup_toolbox(self):
        self.toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            self._gen_individual,
        )
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual,
        )

        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register(
            "select",
            tools.selTournament,
            tournsize=self.CONFIG.get("tournament_size", 3),
        )
        self.toolbox.register("mate", tools.cxOnePoint)

        self.toolbox.register(
            "mutate",
            self._custom_mutation,
            indpb=self.CONFIG.get("indpb", 0.2),
        )

    def _gen_individual(self):
        return [
            random.uniform(self.lb[i], self.ub[i]) for i in range(len(self.var_names))
        ]

    def _evaluate(self, individual):
        t_overrides = {}
        for i, name in enumerate(self.var_names):
            t_overrides[name] = self.var_types[i](individual[i])

        final_overrides = {**t_overrides, **self.static_overrides}

        sim_result = self.run_simulation(final_overrides)

        obj = self.objective_function(
            sim_result["hourly_energy"],
            sim_result["pc_htf_pump_power"],
            sim_result["field_htf_pump_power"],
        )

        return (float(obj[0]),)

    def _custom_mutation(self, individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                if self.var_types[i] is int:
                    individual[i] = random.randint(self.lb[i], self.ub[i])
                else:
                    sigma = 0.1 * (self.ub[i] - self.lb[i])
                    individual[i] += random.gauss(0, sigma)

                individual[i] = max(self.lb[i], min(self.ub[i], individual[i]))

        return (individual,)

    def run(self):
        pop_size = self.CONFIG["pop_size"]
        cxpb = self.CONFIG["cxpb"]
        mutpb = self.CONFIG["mutpb"]
        num_generations = self.CONFIG["num_generations"]

        pop, hof, logbook, start_gen = self._initialize_or_resume(pop_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        if start_gen == 0:
            self._evaluate_population(pop)
            hof.update(pop)

        for gen in range(start_gen, num_generations):
            offspring = algorithms.varAnd(pop, self.toolbox, cxpb, mutpb)
            self._evaluate_population(offspring)

            pop = self.toolbox.select(offspring, k=pop_size)
            hof.update(pop)

            record = stats.compile(pop)
            logbook.record(gen=gen, **record)
            self.mlflow.log_metrics(record, step=gen)

            self._save_checkpoint(gen, pop, hof, logbook)

        self.pool.close()
        self.pool.join()

        return self._finalize(hof)
