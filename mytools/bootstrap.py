import itertools as itl
from itertools import repeat
import multiprocessing as mlp
from collections import namedtuple
import numpy as np
import random
from mytools.multiproc import run_fun_args


def resample(dataset, n_samples=1000):
    dataset = np.asarray(dataset)
    # pull out resampled data for n_samples
    resampled = [None] * n_samples
    for i in range(n_samples):
        resampled[i] = random.choices(dataset, k=len(dataset))
    return np.asarray(resampled)


def bs_stats(resampled_means, name='output'):
    # input resampled means
    Output = namedtuple(name, ['stdev', 'sem', 'ci95', 'mean'],
                        defaults=[np.var(resampled_means, ddof=1), np.std(resampled_means, ddof=1),
                                  np.percentile(resampled_means, [2.5, 97.5]), np.mean(resampled_means)])
    return Output()


def bootstrap_logistic_fit(fun, x_data, y_data, maxfev=500, p0=None, bounds=None, method='trf', use_mp=False):
    # args e.g. (x_vals, y_vals) input as list to be used in next function
    if use_mp:
        x_data = [x_data] * len(y_data)  # make same length for iteration in multiprocessing'
        maxfev = [maxfev] * len(y_data)
        p0 = [p0] * len(y_data)
        bounds = [bounds] * len(y_data)
        method = [method] * len(y_data)
        _bootstrapping = [True] * len(y_data)
        # use multiprocessing with fit function
        with mlp.Pool() as pool:
            # resampling with og data for n_samples (zip keeps items from each list in pairs)
            resamp_data = pool.starmap(fun, zip(x_data, y_data, maxfev, p0, bounds, method, _bootstrapping))
            # resamp_data = pool.starmap(fun, zip(repeat(x_data), y_data, repeat(maxfev), repeat(p0), repeat(bounds),
            #                                     repeat(method), repeat(True)))
            pool.close()
            pool.join()
    else:
        resamp_data = list(itl.starmap(fun, zip(repeat(x_data), y_data, repeat(maxfev), repeat(p0), repeat(bounds),
                                                repeat(method), repeat(True))))

    # Calculate descriptive statistics
    if any(_r is None for _r in resamp_data):
        print(f"{resamp_data.count(None)} failed fits within boostrapping\n"
              f"Removing failed fits from further analysis: WARNING this may influence bs estimates")
        resamp_data = [_r for _r in resamp_data if _r is not None]  # remove failed fits TODO seems wrong
    bootstrap_info = bs_stats(resamp_data, name='threshold')

    return bootstrap_info

