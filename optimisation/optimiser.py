import matlab.engine

from config import CONFIG


def run_optimisation():
    # Start MATLAB
    eng = matlab.engine.start_matlab()
    
    # Add path where your .m file is stored
    eng.addpath(CONFIG["matlab_folder"], nargout=0)

    # fmincon options
    options = eng.optimoptions("fmincon",
                            "Display", CONFIG["display"],
                            "Algorithm", CONFIG["algorithm"])

    # Run fmincon using your external MATLAB functions
    x_opt, fval = eng.fmincon(
        CONFIG["objective_name"],   # MATLAB function inside .m file
        matlab.double(CONFIG["x0"]),
        matlab.double([]), matlab.double([]),   # A, b
        matlab.double([]), matlab.double([]),   # Aeq, beq
        matlab.double(CONFIG["lb"]), matlab.double(CONFIG["ub"]),
        CONFIG["constraint_name"],    # MATLAB function inside same file
        options,
        nargout=2
    )

    # print("Optimal x:", x_opt)
    # print("Optimal value:", fval)

    eng.quit()
    
    return x_opt,fval