import itertools
import logging
from itertools import repeat
import random
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from scipy.linalg import hankel
import pandas as pd
import time
import multiprocessing as mlp
import os
import pylab
from datetime import datetime
import matplotlib.pyplot as plt
from numpy import nan
from .general_functions import run_function_and_unpack_args_kwargs, normalise
from .curve_fit import fit_logistic, plot_psychometric
from components.general_functions import remove_list_repeats
from copy import deepcopy
from mytools.dframe_tools import init_cols
from mytools.list_tools import shuffle
from mytools.analysis import geomean


def set_random_seed():
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))  # random seed for parallel processing


def mocs_task(model, decoder, ori_std, contrast, task='', **kwargs):
    set_random_seed()  # random seed for parallel processing

    # if task not specified then give options
    if all(task != i for i in ['ori_id', 'ori_dis', 'con_det']):
        print("# # # TASK NOT SET # # #\nChoose from:\t'ori_id'\t,\t'ori_dis'\t,\t'con_det'")

    levels = kwargs.get('levels')
    n_lvl = len(levels)
    n_level_reps = kwargs.get('n_level_reps')

    # # # NEW METHOD # # #
    # PROCESS ALL STIMULI AT ONCE #
    if task == 'con_det':
        # Compare response for stimulus at level contrast, and for stimulus at 0 contrast (noise)
        pooled_resp = model.detection(ori_std, stim_con=[*levels, *([0] * n_lvl)], method=decoder, n_reps=n_level_reps)
        probe_resp = pooled_resp[:, :n_lvl]
        null_resp = pooled_resp[:, n_lvl:]
        proportion_correct = np.mean(probe_resp > null_resp, axis=0)
        rawdata = pd.DataFrame({'level': levels, 'propn_corr': proportion_correct})

    elif task == 'ori_dis':
        # Discriminate orientations
        std_oris = [ori_std] * n_lvl
        comp_oris = ori_std + levels

        # All decoded orientations
        decoded = model.decode_response(decoder, [*std_oris, *comp_oris], contrast, n_reps=n_level_reps)

        # Correct for any wrapping errors (circular space) [assumes no realistic estimates below this value]
        min_bound = ori_std - 45  # Bounds of realistic estimates assuming clockwise ori diff
        decoded = np.where(decoded < min_bound, 180 + decoded, decoded)

        std_decoded = decoded[:, :n_lvl]
        comp_decoded = decoded[:, n_lvl:]
        proportion_correct = np.mean(comp_decoded > std_decoded, axis=0)
        rawdata = pd.DataFrame({'level': levels, 'propn_corr': proportion_correct})

    # Determine threshold
    if any(i >= .75 for i in rawdata['propn_corr']):  # Above chance performance
        curvefit = fit_logistic(x=levels, y=rawdata['propn_corr'])

        if kwargs.get('_plot_psychometric') is True:
            if kwargs.get('_psychometric_dir'):
                _save = True
                save_dir = kwargs.get('_psychometric_dir')
            else:
                _save = False
                save_dir = None
            # for tasks where orientation is manipulated
            if any(i == task for i in ['ori_id', 'ori_dis']):
                if task == 'ori_dis':
                    plot_psychometric(x_axis=levels, y_axis=rawdata['propn_corr'],
                                      x_fit_axis=curvefit['x'], y_fit_axis=curvefit['y'],
                                      threshold_value=curvefit['threshold'],
                                      x_label='Orientation Difference (deg)', y_label='Proportion Correct',
                                      y_lim=(0.45, 1.02), x_scale='log', _percent=True,
                                      _save=_save, save_dir=save_dir, save_name=f"{task.upper()}_{decoder}",
                                      title=f"{task.upper()} {ori_std}: {decoder} at {contrast}% contrast")
            # for tasks where orientation is manipulated
            elif task == 'con_det':
                plot_psychometric(x_axis=levels, y_axis=rawdata['propn_corr'],
                                  x_fit_axis=curvefit['x'], y_fit_axis=curvefit['y'],
                                  threshold_value=curvefit['threshold'],
                                  x_label='Contrast (%)', y_label='Proportion Correct',
                                  y_lim=(0.45, 1.02), x_scale='log', _percent=True,
                                  _save=_save, save_dir=save_dir, save_name=f"{task.upper()}_{decoder}",
                                  title=f"{task.upper()}: {ori_std}")
                z = 0#

    else:
        curvefit = {}
        print(f"no fit for Decoder={decoder} Ori={ori_std} Contrast={contrast}")

    results = {'task': task, 'decoder': decoder, 'ori_std': ori_std, 'contrast': contrast,
               **{_k: curvefit.get(_k, nan) for _k in ['threshold', 't_std', 'slope', 's_std']}}

    return pd.DataFrame(results, index=[''])


def run(experiment, model, decoders=('WTA', 'PV', 'ML'), stdoris=(0, 45), contrasts=(20, 40), n_runs=1,
        design='mocs', run_type_='loop', **kwargs):
    dataset = None  # Initialise output container
    decoders = [decoders] if not isinstance(decoders, (list, tuple)) else decoders

    # --- Mode: Single predefined condition ---
    if run_type_ == 'single':
        dataset = experiment(model=model, decoder='WTA', ori_std=45, contrast=100, **kwargs)

    # --- Mode: Loop through all condition combinations ---
    elif run_type_ == 'loop':
        tic = time.time()
        loopruns_data = []

        for _ in range(n_runs):
            for ori_std, decoder, contrast in itertools.product(stdoris, decoders, contrasts):
                run_data = experiment(model=model, decoder=decoder, ori_std=ori_std, contrast=contrast, **kwargs)
                loopruns_data.append(run_data)

        dataset = pd.concat(loopruns_data, ignore_index=True)
        # print(f"Loop runtime: {time.time() - tic:.2f} sec")

    # --- Mode: Parallel execution using multiprocessing ---
    elif run_type_ == 'multi_proc':
        tic = time.time()

        # Build argument sets for multiprocessing
        mp_args = []
        for _ in range(n_runs):
            for ori_std, decoder, contrast in itertools.product(stdoris, decoders, contrasts):
                mp_args.append((model, decoder, ori_std, contrast))

        # Include kwargs for each task
        mp_kwargs = repeat(kwargs)

        # Combine experiment function with zipped args and repeated kwargs
        args_for_starmap = zip(repeat(experiment), mp_args, mp_kwargs)

        try:
            with mlp.get_context('spawn').Pool() as pool:
                multiproc_data = pool.starmap(run_function_and_unpack_args_kwargs, args_for_starmap)
        except Exception as err:
            logging.getLogger(__name__).exception(err)
            raise  # Allow the exception to propagate if needed

        dataset = pd.concat(multiproc_data, ignore_index=True)
        dataset = dataset.sort_values(by=['decoder', 'ori_std', 'contrast'])
        print(f"Multiprocessing runtime: {time.time() - tic:.2f} sec")

    else:
        raise ValueError(f"Invalid run_type_: {run_type_}. Choose from 'single', 'loop', or 'multi_proc'.")

    # --- Append design metadata ---
    dataset['design'] = design

    return dataset

