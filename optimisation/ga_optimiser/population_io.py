import pickle
from pathlib import Path


def save_population(population, gen, metadata):
    out_dir = Path(f"populations/gen_{gen}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, ind in enumerate(population):
        payload = {"individual": list(ind), **metadata}

        with open(out_dir / f"ind_{idx}.pkl", "wb") as f:
            pickle.dump(payload, f)
