function [state, options, optchanged] = save_state_ga(options, state, flag)
optchanged = false;

% Create checkpoint at every generation
if strcmp(flag, 'iter')
    save('ga_checkpoint.mat', 'state', 'options', '-v7.3');
    fprintf('Checkpoint saved at generation %d\n', state.Generation);
end
end
