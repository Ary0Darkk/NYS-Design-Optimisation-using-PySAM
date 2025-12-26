# setup_cache.py
from prefect.filesystems import LocalFileSystem

# Use a relative path so it works across different PC usernames
# This will create a '.prefect_cache' folder in your project
sub_folder = ".prefect_cache"

block = LocalFileSystem(basepath=sub_folder)
block.save("local-storage", overwrite=True)

print(
    f"✅ Success: Block 'local-storage' registered. Results will save to {sub_folder}"
)
