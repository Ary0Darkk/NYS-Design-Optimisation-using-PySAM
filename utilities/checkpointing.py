import os
import pickle
import tempfile

def atomic_pickle_dump(obj, path):
    path = str(path)
    dir_name = os.path.dirname(path)

    with tempfile.NamedTemporaryFile(delete=False, dir=dir_name) as tf:
        pickle.dump(obj, tf)
        temp_name = tf.name

    os.replace(temp_name, path)