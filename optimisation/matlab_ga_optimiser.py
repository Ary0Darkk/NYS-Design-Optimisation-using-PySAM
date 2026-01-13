import mlflow
import dagshub
from config import CONFIG


def run_ga_optimisation():
    try:
        import matlab.engine
    except ImportError:
        raise RuntimeError("MATLAB Engine required for GA mode")

    # database setup
    mlflow.set_tracking_uri(
        "https://dagshub.com/aryanvj787/NYS-Design-Optimisation-using-PySAM.mlflow"
    )
    dagshub.init(
        repo_owner="aryanvj787",
        repo_name="NYS-Design-Optimisation-using-PySAM",
        mlflow=True,
    )

    # set experiment name
    mlflow.set_experiment("matlab-ga-optimisation")

    # SAFETY: close any run that may be active from earlier imports/calls
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # set run name here
    run_name = CONFIG["run_name"]

    # Start MATLAB engine
    eng = matlab.engine.start_matlab()

    with mlflow.start_run(run_name=run_name):
        try:
            # author tag
            mlflow.set_tag("Author", CONFIG["author"])

            # Add path where your .m file is stored
            eng.addpath(CONFIG["matlab_folder"], nargout=0)

            # Convert Python numbers to MATLAB double arrays
            lb_matlab = matlab.double(CONFIG["lb"])
            ub_matlab = matlab.double(CONFIG["ub"])

            print("Running MATLAB GA optimization...")

            # Build fitness function handle
            fitness = eng.eval(f"@{CONFIG['objective_name']}", nargout=1)
            constraints = eng.eval(f"@{CONFIG['constraint_name']}", nargout=1)
            # plot = eng.eval(@gaplotbestf, nargout=1)

            config_vars = {
                "Display": CONFIG["display"],
                "EliteCount": CONFIG["elite_count"],
                # "HybridFcn":CONFIG["hybrid_fcn"],
                # "PlotFcn": plot,
                "MaxGenerations": CONFIG["max_generations"],
                "PopulationSize": CONFIG["pop_size"],
                "UseParallel": CONFIG["use_parallel"],
            }

            options = eng.optimoptions("ga")

            for k, v in config_vars.items():
                options = eng.optimoptions(options, k, v)
            options = eng.optimoptions(options, nargout=1)

            # GA options with logging
            # options = eng.optimoptions("ga",
            #     "Display", CONFIG["display"],
            #     "EliteCount",matlab.int16(CONFIG["elite_count"]),
            #     # "HybridFcn",CONFIG["hybrid_fcn"],
            #     "MaxGenerations",CONFIG["max_generations"],
            #     "PopulationSize",CONFIG["pop_size"],
            #     "UseParallel",CONFIG["use_parallel"],
            #     nargout=1
            #     )

            # mlflow.log_params(config_vars)
            mlflow.log_params(
                {
                    "num_vars": int(len(CONFIG["x0"])),
                    "lb": CONFIG["lb"],
                    "ub": CONFIG["ub"],
                }
            )

            mlflow.log_artifacts(
                "objective_functions", artifact_path="matlab_function_files"
            )

            # FIXME : constraints and options get incorrectly parsed,
            # num of arg issue
            # ----- CORRECT GA CALL -----
            x_opt, fval, exitflag, output, population, scores = eng.ga(
                fitness,
                float(len(CONFIG["overrides"])),
                matlab.double([]),
                matlab.double([]),
                matlab.double([]),
                matlab.double([]),
                lb_matlab,
                ub_matlab,
                matlab.double([]),  # empty nonlcon
                options,
                nargout=6,  # must be OUTSIDE arguments
            )

            # mat-to-python conversion
            x_opt = [x for row in x_opt for x in row]
            x_dict = {}
            for var, value in zip(CONFIG["overrides"], x_opt):
                rav = var + " optimal value"
                x_dict[rav] = float(value)

            # log to mlflow ui
            mlflow.log_metrics(x_dict)
            mlflow.log_metric("Optimal function value", float(-fval))
            mlflow.log_metric("Exit flag", float(exitflag))
            mlflow.log_param("Population", population)
            mlflow.log_param("Scores", scores)
            mlflow.log_params(output)

            print(f"Best x: {x_dict}")
            print(f"Best f(x): {-fval}")

            eng.quit()

            return x_dict, -fval

        finally:
            eng.quit()
