import random
import numpy as np
import mlflow
from deap import base, creator, tools, algorithms

import multiprocessing as mp

from prefect import task
from prefect.logging import get_run_logger

from config import CONFIG

# individual creation
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


class GAEngine:
    def __init__(
        self,
        override,
        static_overrides,
    ):
        self.override = override
        self.var_names = self.override["var_names"]
        self.var_types = self.override["var_types"]
        self.lb = self.override["lb"]
        self.ub = self.override["ub"]
        self.static_overrides = static_overrides

        self.pool = mp.Pool(processes=self.CONFIG.get("n_jobs", mp.cpu_count()))

        self.toolbox = base.Toolbox()
        self.toolbox.register("map", self.pool.map)
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
            tournsize=self.CONFIG.get("tournament_size"),
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

    def _evaluate(self, individual, hour):
        t_overrides = {}
        for i, name in enumerate(self.var_names):
            t_overrides[name] = self.var_types[i](individual[i])

        final_overrides = {**t_overrides, **self.static_overrides}

        sim_result = self.run_simulation(final_overrides)

        obj = self.objective_function(
            sim_result["hourly_energy"],
            sim_result["pc_htf_pump_power"],
            sim_result["field_htf_pump_power"],
            hour_index=hour,
        )
        fitness = float(obj)

        return (fitness,)

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

    def run(self, is_nested, curr_hour):
        self.logger = get_run_logger()

        if mlflow.active_run() and not is_nested:
            mlflow.end_run()

        with mlflow.start_run(run_name=f"GA_hour_{curr_hour}", nested=is_nested):
            mlflow.set_tag("Author", self.CONFIG["author"])
            mlflow.log_artifact("config.py")

            pop_size = CONFIG["pop_size"]
            cxpb = CONFIG["cxpb"]
            mutpb = CONFIG["mutpb"]
            num_generations = CONFIG["num_generations"]

            self.logger.info("GA started!")

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


@task()
def run_deap_ga(
    override: dict,
    static_overrides: dict[str, float],
    is_nested: bool,
    curr_hour: int,
):
    # class instantiation
    ga = GAEngine(override, static_overrides, curr_hour)

    # runs ga
    result = ga.run(is_nested, curr_hour)

    return result
