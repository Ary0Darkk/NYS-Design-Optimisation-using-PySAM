import mlflow
import requests
from dagshub import init


def initialize_mlflow(repo_owner, repo_name):
    """
    Attempts to connect to DagsHub MLflow.
    Falls back to local SQLite if unreachable.
    """
    # DagsHub specific URI format
    remote_url = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    local_uri = "sqlite:///mlflow.db"

    try:
        # 1. Check internet/DagsHub connectivity
        response = requests.get(
            f"https://dagshub.com/{repo_owner}/{repo_name}", timeout=3
        )

        if response.status_code == 200:
            print(f"Connected to DagsHub: {repo_owner}/{repo_name}")

            # 2. This helper function handles auth (Username/Token) for you!
            # It sets the MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD env vars.
            init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)

            mlflow.set_tracking_uri(remote_url)
        else:
            raise ConnectionError

    except (requests.exceptions.RequestException, ConnectionError):
        print("DagsHub unreachable. Switching to Local SQLite.")
        mlflow.set_tracking_uri(local_uri)


# --- Usage ---
# Replace with your actual DagsHub username and repository name
# initialize_mlflow(repo_owner="your_username", repo_name="your_repo")

# with mlflow.start_run():
#     mlflow.log_param("mode", "dagshub_test")
#     mlflow.log_metric("accuracy", 0.95)
