import matlab.engine
import mlflow
from config import CONFIG
from convert import convert

# database setup
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

# set experiment name
mlflow.set_experiment("fmincon-optimisation")

# set run name here
run_name = None

def run_fmincon_optimisation():
    # Start MATLAB
    eng = matlab.engine.start_matlab()
    
    with mlflow.start_run(run_name=run_name):
    
        # Add path where your .m file is stored
        eng.addpath(CONFIG["matlab_folder"], nargout=0)
        
        # print
        print('Running fmincon optimiser...')
        
        config_vars = {
            "Display": CONFIG["display"],
            "Algorithm": CONFIG["algorithm"],
            "MaxFunctionEvaluations":CONFIG["max_function_evaluation"],
            "MaxIterations": CONFIG["max_iterations"],
            "ConstraintTolerance":CONFIG["constraint_tolerance"],
        }

        options = eng.optimoptions("fmincon")

        for k, v in config_vars.items():
            options = eng.optimoptions(options, k, v)

        # fmincon options
        # options = eng.optimoptions("fmincon",
        #                         "Display", CONFIG["display"],
        #                         "Algorithm", CONFIG["algorithm"],
        #                         "MaxFunctionEvaluations",CONFIG["max_function_evaluation"],
        #                         "MaxIterations",CONFIG["max_iterations"],
        #                         "ConstraintTolerance",CONFIG["constraint_tolerance"])
        
        
        mlflow.log_params(config_vars)
        mlflow.log_params({
            "num_inputs": int(len(CONFIG["x0"])),
            "lb": CONFIG["lb"],
            "ub": CONFIG["ub"],
        })
        
        mlflow.log_artifacts("objective_functions",artifact_path="matlab_function_files")

        # Run fmincon using your external MATLAB functions
        x_opt, fval, exitflag, output, lam, grad, hessian = eng.fmincon(
            CONFIG["objective_name"],   # MATLAB function inside .m file
            matlab.double(CONFIG["x0"]),
            matlab.double([]), matlab.double([]),   # A, b
            matlab.double([]), matlab.double([]),   # Aeq, beq
            matlab.double(CONFIG["lb"]), matlab.double(CONFIG["ub"]),
            CONFIG["constraint_name"],    # MATLAB function inside same file
            options,
            nargout=7
        )
        
        # mat-to-python conversion
        x_opt = [x for row in x_opt for x in row]
        x_dict={}
        for var,value in zip(CONFIG["overrides"],x_opt):
            rav = var + " optimal value"
            x_dict[rav] = float(value)
        
        # log to mlflow ui
        mlflow.log_metrics(x_dict)
        mlflow.log_metric('Optimal function value',float(fval))
        mlflow.log_metric('Exit flag',float(exitflag))
        mlflow.log_param('Hessian',hessian)
        mlflow.log_param('Gradient',grad)
        mlflow.log_params(output)
        mlflow.log_params(lam)
        
        
        print("Optimal x:", x_dict)
        print("Optimal function value:", fval)

        eng.quit()
        
        return x_opt,fval