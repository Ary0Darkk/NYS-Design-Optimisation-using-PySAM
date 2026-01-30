import httpx
from router import run_router
from utilities.setup_custom_logger import setup_custom_logger


def main():
    # Initialize the file logger with the Run ID
    logger = setup_custom_logger()
    logger.info("NYS-Optimisation started!")
    # runs the router
    run_router()
    logger.info("Optimisation completed!")


if __name__ == "__main__":
    try:
        # proceed with normal run using modified CONFIG
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by User! Shutting down workers...")
    except (httpx.ConnectError, ConnectionError) as err:
        print('f"CRITICAL: Could not connect to Prefect Server error!')
        print(f"Error : {err}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
    finally:
        print("Optimisation completed! /nCheck the mlflow UI for details!")
