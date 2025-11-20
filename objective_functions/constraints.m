% constraints
function [c,ceq] = constraints(x)
% CONSTRAINTS_PYSAM – nonlinear constraints for optimization.
% You can customize constraints from config.py as well.

% Load config.py
cfg = py.config.CONFIG;

% Example constraint from config (optional)
% max T_startup and T_shutdown allowed
max_T_startup  = double(cfg{'lb'}{1});
max_T_shutdown = double(cfg{'ub'}{2});

% Example inequality constraints:
% c(x) <= 0

c = [
    x(1) - max_T_startup;    % T_startup <= max allowed
    x(2) - max_T_shutdown;   % T_shutdown <= max allowed
    -(x(1));                 % T_startup >= 0
    -(x(2));                 % T_shutdown >= 0
    ];

% No equality constraints
ceq = [];
end