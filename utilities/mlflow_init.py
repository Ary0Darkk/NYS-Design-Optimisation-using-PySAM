import os
import mlflow
import requests
from dagshub import init


def initialize_mlflow(repo_owner, repo_name):
    remote_url = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    local_uri = "sqlite:///mlflow.db"

    # Check if we are already authorized via environment variables (from PBS script)
    has_auth_vars = os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv(
        "MLFLOW_TRACKING_PASSWORD"
    )

    try:
        # 1. Quick connectivity check
        requests.get("https://dagshub.com", timeout=2)

        print(f"Connected to DagsHub: {repo_owner}/{repo_name}")

        # 2. Only call init() if we DON'T have environment variables set
        if not has_auth_vars:
            init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)

        mlflow.set_tracking_uri(remote_url)

    except (requests.exceptions.RequestException, ConnectionError):
        print("DagsHub unreachable or No Internet. Switching to Local SQLite.")
        mlflow.set_tracking_uri(local_uri)
