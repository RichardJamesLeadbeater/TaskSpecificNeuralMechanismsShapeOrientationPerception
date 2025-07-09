import numpy as np
import os


def make_paths(paths):
    paths = list(np.atleast_1d(paths))
    for i_path in paths:
        if not os.path.exists(i_path):
            os.makedirs(i_path)
