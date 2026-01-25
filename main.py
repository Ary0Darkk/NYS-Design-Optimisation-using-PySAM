import httpx
from router import run_router
from utilities.setup_custom_logger import setup_custom_logger


# def load_repro_config(path):
#     """Load config (config.py-style dict)"""
#     if path.endswith(".py"):
#         # risky but works: execute file and capture CONFIG
#         ns = {}
#         exec(open(path).read(), ns)
#         return ns.get("CONFIG", None)
#     else:
#         raise ValueError("Unsupported repro config format")


# def find_latest_downloaded_config():
#     files = sorted(
#         glob.glob("downloaded_run_artifacts/*.py"),
#         key=os.path.getmtime,
#         reverse=True,
#     )
#     if not files:
#         raise FileNotFoundError(
#             "No downloaded config found in downloaded_run_artifacts/"
#         )
#     return files[0]


def main():
    # Initialize the file logger with the Run ID
    logger = setup_custom_logger()
    logger.info("NYS-Optimisation started!")
    # runs the router
    run_router()
    logger.info("Optimisation completed!")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--repro",
    #     nargs="?",
    #     const="AUTO",
    #     help="Reproduce using downloaded config. Use AUTO for latest.",
    # )
    # args = parser.parse_args()

    # # override CONFIG if repro mode is used
    # if args.repro:
    #     if args.repro == "AUTO":
    #         repro_file = find_latest_downloaded_config()
    #         print(f"[REPRO] Auto-loading latest downloaded config: {repro_file}")
    #     else:
    #         repro_file = args.repro
    #         print(f"[REPRO] Loading specified config: {repro_file}")

    #     repro_cfg = load_repro_config(repro_file)
    #     if repro_cfg is None:
    #         raise RuntimeError("Failed to load CONFIG from repro file.")

    #     print("[REPRO] Overriding CONFIG with loaded values...")
    #     CONFIG.update(repro_cfg if "CONFIG" not in repro_cfg else repro_cfg["CONFIG"])

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
