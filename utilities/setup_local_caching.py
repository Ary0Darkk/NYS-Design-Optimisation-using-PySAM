import os
from prefect.filesystems import LocalFileSystem
from prefect.settings import PREFECT_API_URL, update_current_profile


def initialize_local_prefect():
    print("Starting Local Prefect Setup...")

    # 1. Force Prefect to use the local server
    # This ensures your code doesn't try to talk to Prefect Cloud
    local_url = "http://127.0.0.1:4200/api"
    update_current_profile(settings={PREFECT_API_URL: local_url})
    print(f"API URL set to: {local_url}")

    # 2. Create the local results directory
    result_path = os.path.abspath("./prefect-results")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Created directory: {result_path}")

    # 3. Register the Storage Block
    # This fixes the "Result storage configuration must be persisted" error
    local_storage = LocalFileSystem(basepath=result_path)
    local_storage.save("local-storage", overwrite=True)
    print("Registered block: local-file-system/local-storage")

    # 4. Create a dummy 's3-cache' block locally
    # This prevents the 404 error if your code still references 's3-cache'
    # We point it to the local path instead of actual S3
    s3_fake_block = LocalFileSystem(basepath=result_path)
    s3_fake_block.save("s3-cache", overwrite=True)
    print("Registered dummy block: local-file-system/s3-cache (redirected to local)")

    print("\nSetup Complete! You can now run 'uv run py main.py'")


if __name__ == "__main__":
    initialize_local_prefect()
