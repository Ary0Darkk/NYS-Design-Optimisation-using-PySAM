import torch
import mlflow
import numpy as np
from pathlib import Path

from rl_reinforce_opt_env import GaussianSearchPolicy, evaluate_candidate
from config import CONFIG


def train_reinforce_opt(
    override,
    optim_mode,
    static_overrides,
    hour_index,
    is_nested=False,
):
    var_names = override["overrides"]
    var_types = override["types"]
    lb, ub = np.array(override["lb"]), np.array(override["ub"])
    dim = len(var_names)

    timestamp = CONFIG["session_time"]

    run_name = (
        "REINFORCE_Design" if optim_mode == "design" else f"REINFORCE_hour_{hour_index}"
    )

    if mlflow.active_run() and not is_nested:
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name, nested=is_nested):
        mlflow.set_tag("optimiser", "REINFORCE-OPT")
        mlflow.set_tag("hour", hour_index)

        # ---- Policy ----
        policy = GaussianSearchPolicy(dim).to("cpu")
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=CONFIG["rl_lr"],
        )

        baseline = None
        beta = 0.9  # EMA baseline

        n_iters = CONFIG["rl_timesteps"]
        batch_size = CONFIG["rl_batch_size"]

        best_reward = -np.inf
        best_params = None

        for it in range(n_iters):
            x, log_prob = policy.sample(batch_size)

            # bounds
            x = torch.clamp(
                x,
                torch.tensor(lb),
                torch.tensor(ub),
            )

            rewards = []
            for xi in x:
                xi_np = xi.detach().numpy()

                # mixed variables
                for i, t in enumerate(var_types):
                    if t == "int":
                        xi_np[i] = int(round(xi_np[i]))

                r = evaluate_candidate(
                    xi_np,
                    optim_mode,
                    hour_index,
                    static_overrides,
                    var_names,
                )
                rewards.append(r)

                if r > best_reward:
                    best_reward = r
                    best_params = dict(zip(var_names, xi_np))

            rewards = torch.tensor(rewards, dtype=torch.float32)

            # baseline
            if baseline is None:
                baseline = rewards.mean()
            else:
                baseline = beta * baseline + (1 - beta) * rewards.mean()

            advantage = rewards - baseline
            loss = -(advantage.detach() * log_prob).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % 10 == 0:
                mlflow.log_metric("mean_reward", rewards.mean().item(), step=it)
                mlflow.log_metric("best_reward", best_reward, step=it)

        # ---- Save results ----
        result_dir = Path(f"results/RL/REINFORCE_{timestamp}")
        result_dir.mkdir(parents=True, exist_ok=True)

        mlflow.log_metric("final_best_reward", best_reward)
        for k, v in best_params.items():
            mlflow.log_param(f"opt_{k}", v)

        return best_params, best_reward
