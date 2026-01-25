import os
from pathlib import Path
import pickle
import tempfile
import logging

logger = logging.getLogger("NYS_Optimisation")


def atomic_pickle_dump(obj, path):
    path = Path(path)
    dir_name = path.parent
    temp_name = None

    try:
        # Create temp file in the same directory to ensure atomic rename capability
        with tempfile.NamedTemporaryFile(
            delete=False, dir=dir_name, suffix=".tmp"
        ) as tf:
            temp_name = tf.name
            pickle.dump(obj, tf)

        # Atomically replace the old file with the new one
        os.replace(temp_name, str(path))
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        if temp_name and os.path.exists(temp_name):
            os.remove(temp_name)
        raise e
