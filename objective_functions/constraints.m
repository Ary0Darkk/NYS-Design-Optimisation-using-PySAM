function [c, ceq] = constraints(x)
% DYNAMIC NONLINEAR CONSTRAINTS FOR MATLAB OPTIMIZATION
% Reads variable names, lower bounds, upper bounds from config.py dynamically.

% Load config.py
cfg = py.config.CONFIG;

% Python lists
var_names = cfg{"overrides"};
lb_list   = cfg{"lb"};
ub_list   = cfg{"ub"};

n = length(var_names);

% Initialize constraint vector
c = zeros(2*n, 1);   % two constraints per variable
idx = 1;

% Build constraints dynamically
for i = 1:n
    lb_i = double(lb_list{i});   % lower bound of variable i
    ub_i = double(ub_list{i});   % upper bound of variable i

    xi = x(i);

    % Inequality constraints:
    % 1) x(i) >= lb(i)   →   lb(i) - x(i) <= 0
    % 2) x(i) <= ub(i)   →   x(i) - ub(i) <= 0

    c(idx)   = lb_i - xi;   % lower bound constraint
    c(idx+1) = xi - ub_i;   % upper bound constraint

    idx = idx + 2;
end

% No equality constraints
ceq = [];
end