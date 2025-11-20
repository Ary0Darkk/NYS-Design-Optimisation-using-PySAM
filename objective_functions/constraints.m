% constraints
function [c, ceq] = constraints(x)
    c = x(1)^2 + x(2)^2 - 4;
    ceq = [];
end