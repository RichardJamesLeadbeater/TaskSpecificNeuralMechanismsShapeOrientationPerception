import multiprocessing as mlp
import numpy as np


def run_fun_args(fun, args):
    # Use multiprocesing to run a function
    # Assumes args is a list of values in the same order as function arguments
    with mlp.Pool() as pool:
        out_data = pool.starmap(fun, zip(*args))
        pool.close()
        pool.join()
    z = 0
    return out_data

