function stop = save_state_fmincon(x, optimValues, state)
stop = false;
if strcmp(state, 'iter')
    lambda_vals = [];
    if isfield(optimValues, 'lambda')
        lambda_vals = optimValues.lambda;
    end
    save(CHECKPOINT_FILE, 'x', 'lambda_vals');
end
end