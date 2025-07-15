import itertools
import logging
from itertools import repeat
import random
import numpy as np
from scipy.linalg import hankel
import pandas as pd
import time
import dill
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


def run(experiment, model, decoders=('WTA', 'PV', 'ML'), stdoris=(0, 45), contrasts=(20, 40), n_runs=1,
        design='mocs', run_type_='loop', **kwargs):
    dataset = None  # Initialise output container

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


def random_choice(vals, n=None):
    # Random sampling WITH replacement
    vals = np.asarray(vals)
    chosen_val = vals[np.random.randint(0, len(vals), n)]
    return chosen_val


def set_random_seed():
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))  # random seed for parallel processing


def mocs_task(model, decoder, ori_std, contrast, task='', **kwargs):
    set_random_seed()  # random seed for parallel processing

    # if task not specified then give options
    if all(task != i for i in ['ori_dis', 'con_det']):
        print("# # # TASK NOT SET # # #\nChoose from:\t'ori_dis'\t,\t'con_det'")

    levels_ = kwargs.get('levels')
    n_level_reps_ = kwargs.get('n_level_reps')

    # run method of constant stimuli
    rawdata = {'level': [], 'propn_corr': []}
    for i_level in levels_:  # loop through each level

        # Orientation Discrimination
        if task == 'ori_dis':  # for tasks where orientation is manipulated
            # HALF CW and HALF CCW (prevent WTA bias at low N attracted to nearest channel)
            cw_comp_ori = ori_std + i_level  # CW rotation
            ccw_comp_ori = ori_std - i_level  # CCW rotation
            split_n_reps = int(n_level_reps_ / 2)
            # DECODE ORIENTATION (slight noise added in model.decode_response())
            # Clockwise
            cw_std_decoded = model.decode_response(decoder, ori_std, contrast, split_n_reps)
            cw_comp_decoded = model.decode_response(decoder, cw_comp_ori, contrast, split_n_reps)  # CLOCKWISE
            # Counterclockwise
            ccw_std_decoded = model.decode_response(decoder, ori_std, contrast, split_n_reps)
            ccw_comp_decoded = model.decode_response(decoder, ccw_comp_ori, contrast, split_n_reps)  # CCW
            # DECISION
            if task == 'ori_dis':
                cw_is_corr = cw_comp_decoded > cw_std_decoded  # Clockwise
                ccw_is_corr = ccw_comp_decoded < ccw_std_decoded  # Counterclockwise
                is_corr = np.hstack([cw_is_corr, ccw_is_corr])  # Proportion correct
            elif task == 'ori_id':
                # Generate template orientation
                template_ori_cw = model.gen_template_ori(ori_std, split_n_reps)
                template_ori_ccw = model.gen_template_ori(ori_std, split_n_reps)
                # Clockwise
                cw_is_corr = abs(cw_std_decoded - template_ori_cw) < abs(cw_comp_decoded - template_ori_cw)
                ccw_is_corr = abs(ccw_std_decoded - template_ori_ccw) < abs(ccw_comp_decoded - template_ori_ccw)
                # Proportion correct
                is_corr = np.hstack([cw_is_corr, ccw_is_corr])
            else:
                is_corr = None
        elif task == 'con_det':
            # compare response for stimulus at level contrast, and for stimulus at 0 contrast (noise)
            std_resp = model.detection(decoder, ori_std, stim_con=i_level, n_reps=n_level_reps_)
            comp_resp = model.detection(decoder, ori_std, stim_con=0, n_reps=n_level_reps_)
            is_corr = std_resp > comp_resp
        propn_corr = (1 / len(is_corr)) * np.sum(is_corr)  # calc propn corr from all trials at this level
        # append level and propn_corr vals into data_dict
        rawdata['level'].append(i_level)
        rawdata['propn_corr'].append(propn_corr)

    if any(i >= .75 for i in rawdata['propn_corr']):  # only calc threshold if above chance performance
        curvefit = fit_logistic(x=levels_, y=rawdata['propn_corr'])

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
                    plot_psychometric(x_axis=levels_, y_axis=rawdata['propn_corr'],
                                      x_fit_axis=curvefit['x'], y_fit_axis=curvefit['y'],
                                      threshold_value=curvefit['threshold'],
                                      x_label='Orientation Difference (deg)', y_label='Proportion Correct',
                                      y_lim=(0.45, 1.02), x_scale='log', _percent=True,
                                      _save=_save, save_dir=save_dir, save_name=f"{task.upper()}_{decoder}",
                                      title=f"{task.upper()} {ori_std}: {decoder} at {contrast}% contrast")
            # for tasks where orientation is manipulated
            elif task == 'con_det':
                plot_psychometric(x_axis=levels_, y_axis=rawdata['propn_corr'],
                                  x_fit_axis=curvefit['x'], y_fit_axis=curvefit['y'],
                                  threshold_value=curvefit['threshold'],
                                  x_label='Contrast (%)', y_label='Proportion Correct',
                                  y_lim=(0.45, 1.02), x_scale='log', _percent=True,
                                  _save=_save, save_dir=save_dir, save_name=f"{task.upper()}_{decoder}",
                                  title=f"{task.upper()}: {ori_std}")
                z = 0#

        summarydata = {'task': task, 'decoder': decoder, 'ori_std': ori_std, 'contrast': contrast,
                       'threshold': curvefit['threshold'], 't_std': curvefit['threshold_std'],
                       'slope': curvefit['slope'], 's_std': curvefit['slope_std']}

        return pd.DataFrame(summarydata, index=[''])

    else:
        summarydata = {'task': task, 'decoder': decoder, 'ori_std': ori_std, 'contrast': contrast,
                       'threshold': nan, 't_std': nan,
                       'slope': nan, 's_std': nan}
        print(f"Warning: No fit for Decoder={decoder} Ori={ori_std} Contrast={contrast}")

        return pd.DataFrame(summarydata, index=[''])

