import asyncio
import logging
from prefect import get_client
from prefect.client.schemas.filters import LogFilter, LogFilterFlowRunId
from pathlib import Path


async def export_all_logs():
    log_dir = Path("exported_logs")
    log_dir.mkdir(exist_ok=True)

    async with get_client() as client:
        flow_runs = await client.read_flow_runs()
        print(f"Found {len(flow_runs)} flow runs. Exporting...")

        for run in flow_runs:
            log_filter = LogFilter(flow_run_id=LogFilterFlowRunId(any_=[run.id]))
            logs = await client.read_logs(log_filter=log_filter)

            file_name = f"{run.name}_{str(run.id)[:8]}.log"
            file_path = log_dir / file_name

            with open(file_path, "w", encoding="utf-8") as f:
                for log in logs:
                    # Convert the integer level (e.g., 20) to a string (e.g., 'INFO')
                    level_name = logging.getLevelName(log.level)
                    f.write(f"{log.timestamp} | {level_name} | {log.message}\n")

            print(f"Saved: {file_name}")


if __name__ == "__main__":
    asyncio.run(export_all_logs())
