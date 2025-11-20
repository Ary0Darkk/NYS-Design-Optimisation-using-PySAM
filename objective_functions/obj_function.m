function y = obj_function(x)
% OBJECTIVE_PYSAM  Calls Python to compute annual energy using PySAM
% and returns NEGATIVE energy (because fmincon MINIMIZES).

% Import config.py
cfg = py.config.CONFIG;

% Build Python overrides dict
% Example: x(1) = T_startup, x(2) = T_shutdown
overrides = py.dict();
overrides{'T_startup'}  = x(1);
overrides{'T_shutdown'} = x(2);

% Call Python function run_trough(json_file, overrides)
annual_energy = py.simulation.simulation.run_simulation(cfg{'json_file'}, overrides);

% Convert Python float → MATLAB double
annual_energy = double(annual_energy);


% fmincon minimizes → return negative of energy
y = -annual_energy;
end