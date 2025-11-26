function y = obj_function(x)

% Load config.py
cfg = py.config.CONFIG;

% Python list of variable names
var_names = cfg{"overrides"};

% Create empty Python dict
overrides = py.dict();

% Loop through each variable name in config
for i = 1:length(var_names)
    key = char(var_names{i});   % MATLAB → Python string
    overrides{key} = x(i);      % assign x(i) to that key
end

% Call Python simulation
sim_output = py.simulation.simulation.run_simulation(overrides);

y = -double(sim_output); % negate for maximization
end