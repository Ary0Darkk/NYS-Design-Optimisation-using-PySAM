from stable_baselines3.common.callbacks import EvalCallback
import optuna


class TrialEvalCallback(EvalCallback):
    def __init__(self, eval_env, trial, eval_freq=2000, **kwargs):
        super().__init__(eval_env, eval_freq=eval_freq, **kwargs)
        self.trial = trial
        self.eval_idx = 0

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            # Report the score to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Kill trial if it's performing worse than the median trial
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return continue_training
