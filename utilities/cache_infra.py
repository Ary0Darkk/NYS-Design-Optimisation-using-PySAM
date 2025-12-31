# setup_cache.py
from prefect.filesystems import LocalFileSystem
from pathlib import Path

# Google Drive base path (Google Drive Desktop must be installed)
GDRIVE_BASE = Path.home() / "My Drive (1)"

# Prefect cache directory inside Drive
cache_dir = GDRIVE_BASE / "prefect_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

block = LocalFileSystem(basepath=str(cache_dir))
block.save("gdrive-storage", overwrite=True)

print(f"Success: Block 'gdrive-storage' registered. Results will save to {cache_dir}")
