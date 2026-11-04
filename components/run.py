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


def random_choice(vals, n=None):
    # Random sampling WITH replacement
    vals = np.asarray(vals)
    chosen_val = vals[np.random.randint(0, len(vals), n)]
    return chosen_val


def set_random_seed():
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))  # random seed for parallel processing


def mocs_task(model, decoder, ori_std, contrast, task='', **kwargs):

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
        probe_resp = pooled_resp[:n_lvl, :]
        null_resp = pooled_resp[n_lvl:, :]
        is_corr = probe_resp > null_resp

    elif task in ['ori_dis', 'ori_id']:
        # #!flag
        # levels = np.linspace(0, 45, 15)

        # Discriminate orientations
        std_oris = [ori_std] * n_lvl

        if ori_std >= 0:
            COMPARISON_ROTATION = 'CCW'
            comp_oris = ori_std - levels  # CCW rotation prevents edge effects for PV decoder (e.g. when stim_ori nears tiling edge 90)
        else:
            COMPARISON_ROTATION = 'CW'
            comp_oris = ori_std + levels

        # All decoded orientations
        decoded = model.decode_response(decoder, [*std_oris, *comp_oris], contrast, n_reps=n_level_reps,
                                        window=kwargs.get('window', False),
                                        prior=kwargs.get('prior')
                                        )

        # Apply row-wise wrapping correction
        # Correct for any wrapping errors (circular space) [assumes no realistic estimates below this value]
        min_bound = np.asarray([*std_oris, *comp_oris]) - 90  # Bounds of realistic estimates assuming clockwise ori diff
        decoded = np.where(decoded < min_bound[:, None], 180 + decoded, decoded)

        std_decoded = decoded[:n_lvl, :]
        comp_decoded = decoded[n_lvl:, :]

        std_response = model.resp_noisy[:, :n_lvl, :]
        comp_response = model.resp_noisy[:, n_lvl:, :]

        # # SIGNED CIRCULAR ERROR (check for bias in decoded orientations)
        # def ori_error(decoded, true_ori):
        #     return (decoded - true_ori + 90) % 180 - 90
        #
        # comp_err = ori_error(comp_decoded, comp_oris[:, None])
        # comp_bias = np.mean(comp_err, axis=1)
        # for ori, bias in zip(comp_oris, comp_bias):
        #     print(f"{ori:.1f}°: bias = {bias:+.2f}°")

        if task == 'ori_dis':
            if COMPARISON_ROTATION == 'CW':
                is_corr = comp_decoded > std_decoded
            else:
                is_corr = comp_decoded < std_decoded

        elif task == 'ori_id':

            id_decision = 'template_decoded'  # 'template_decoded', 'similarity', 'LLR'

            if id_decision == 'LLR':
                # COMPARING HYPOTHESES OF TARGET VS NON-TARGET
                # 2. Compare with individual decoded comparison orientations
                llr_std = model.target_vs_nontarget_log_L_ratio(std_response, target_oris=ori_std,
                                                                nontarget_oris=std_decoded,
                                                                match_nontarget_and_resp_oris=True)

                llr_comp = model.target_vs_nontarget_log_L_ratio(comp_response, target_oris=ori_std,
                                                                 nontarget_oris=comp_decoded,
                                                                 match_nontarget_and_resp_oris=True)

                # What's more likely to cause template response, decoded template ori or comparison?
                decoded_template = model.template_model.decode_response(
                    decoder, std_oris, contrast, n_reps=n_level_reps, window=False
                )
                template_resp = model.template_model.resp_noisy
                llr_template_comp = model.target_vs_nontarget_log_L_ratio(
                    template_resp, target_oris=decoded_template, nontarget_oris=comp_decoded,
                    match_nontarget_and_resp_oris=True
                )
                llr_template_std = model.target_vs_nontarget_log_L_ratio(
                    template_resp, target_oris=decoded_template, nontarget_oris=std_decoded,
                    match_nontarget_and_resp_oris=True
                )

                is_corr = llr_std > llr_comp
                # is_corr = llr_comp < 0

            elif id_decision == 'similarity':
                    # COSINE SIMILARITY BETWEEN ACTIVITY PATTERNS RATHER THAN EXPLICITLY DECODING ORIENTATIONS
                    _template_type = 'noiseless'  # 'reactivation'

                    if _template_type == 'reactivation':
                        template_response_1 = model.template_model.gen_stim_response(
                            std_oris, model.global_params['template_contrast'], n_level_reps
                        )
                        template_response_2 = model.template_model.gen_stim_response(
                            std_oris, model.global_params['template_contrast'], n_level_reps
                        )
                    elif _template_type == 'noiseless':
                        model.template_model.gen_stim_response(
                            std_oris, model.global_params['template_contrast'], n_level_reps
                        )
                        template_response_1 = model.template_model.resp_tuned + model.param_distribs['spont'][:, None, None]
                        template_response_2 = model.template_model.resp_tuned + model.param_distribs['spont'][:, None, None]

                    _similarity_metric = 'l2'

                    # L1 difference: sum of absolute neuron-by-neuron differences.
                    if _similarity_metric == 'l1':
                        # std_l1_distance = np.sum(np.abs(std_response - template_response_1), axis=0)
                        # comp_l1_distance = np.sum(np.abs(comp_response - template_response_2), axis=0)

                        # Mean absolute difference across neurons
                        std_l1_distance = np.mean(np.abs(std_response - template_response_1), axis=0)
                        comp_l1_distance = np.mean(np.abs(comp_response - template_response_2), axis=0)

                        # Convert distance to similarity
                        std_similarity = 1 / (1 + std_l1_distance)
                        comp_similarity = 1 / (1 + comp_l1_distance)

                    # L2 difference: Euclidean distance between the two response vectors.
                    elif _similarity_metric == 'l2':
                        std_l2_distance = np.linalg.norm(std_response - template_response_1, axis=0)
                        comp_l2_distance = np.linalg.norm(comp_response - template_response_2, axis=0)

                        # Convert distance to similarity
                        std_similarity = 1 / (1 + std_l2_distance)
                        comp_similarity = 1 / (1 + comp_l2_distance)

                    # Compare whole neural activation patterns, across neurons.
                    else:
                        std_similarity = np.sum(std_response * template_response_1, axis=0) / (
                                np.linalg.norm(std_response, axis=0) *
                                np.linalg.norm(template_response_1, axis=0)
                        )
                        comp_similarity = np.sum(comp_response * template_response_2, axis=0) / (
                                np.linalg.norm(comp_response, axis=0) *
                                np.linalg.norm(template_response_2, axis=0)
                        )

                    is_corr = std_similarity > comp_similarity

            elif id_decision == 'template_decoded':
                reactivate_template = True
                if reactivate_template:
                    # Generate template orientation
                    template_ori = model.template_model.decode_response(
                        decoder, std_oris, model.global_params['template_contrast'], n_reps=n_level_reps,
                        prior=kwargs.get('prior'), window=False
                    )
                    # Which stimulus was closest to the template orientation (internall generated)
                    is_corr = abs(std_decoded - template_ori) < abs(comp_decoded - template_ori)

            else:
                is_corr = None

    proportion_correct = np.mean(is_corr, axis=1)  # Proportion correct
    rawdata = pd.DataFrame({'level': levels, 'propn_corr': proportion_correct})

    # Determine threshold
    if any(i >= .75 for i in proportion_correct):  # Above chance performance
        curvefit = fit_logistic(x=levels, y=proportion_correct)

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

    if curvefit.get('threshold'):
        if curvefit.get('threshold') < 0.07:
            print(curvefit.get('threshold'))
    else:
        f"\n\n\n{'=' * 30}\n\n\nNO THRESHOLD SET\n\n\n{'=' * 30}\n\n\n"
        z = 0

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


