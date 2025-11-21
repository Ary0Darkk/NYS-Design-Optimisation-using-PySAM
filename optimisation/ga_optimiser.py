import matlab.engine
from config import CONFIG

def run_ga_optimisation():
    
    # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    
    # Add path where your .m file is stored
    eng.addpath(CONFIG["matlab_folder"], nargout=0)

    # Convert Python numbers to MATLAB double arrays
    lb_matlab = matlab.double(CONFIG["lb"])
    ub_matlab = matlab.double(CONFIG["ub"])

    print("Running MATLAB GA optimization...")

    # Build fitness function handle
    fitness = eng.eval(f"@{CONFIG['objective_name']}", nargout=1)
    constraints = eng.eval(f"@{CONFIG['constraint_name']}", nargout=1)

    # GA options with logging
    options = eng.gaoptimset("Display", "iter",
                             "EliteCount",CONFIG["elite_count"],
                             "HybridFcn",CONFIG["hybrid_fcn"],
                             "MaxGenerations",CONFIG["max_generations"],
                             "PopulationSize",CONFIG["pop_size"],
                             "UseParallel",CONFIG["use_parallel"], 
                             nargout=1)
    
    
    # FIXME : constraints and options get incorrectly parsed,
    # num of arg issue
    # ----- CORRECT GA CALL -----
    x, fval = eng.ga(
        fitness,
        float(len(CONFIG["overrides"])),
        matlab.double([]), matlab.double([]),
        matlab.double([]), matlab.double([]),
        lb_matlab,
        ub_matlab,
        matlab.double([]),  # empty nonlcon
        options,
        nargout=2        # must be OUTSIDE arguments
    )

    print(f"Best x: {x}")
    print(f"Best f(x): {fval}")

    eng.quit()
    
    return x, -fval
