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


def random_choice(vals, n=None):
    # Random sampling WITH replacement
    vals = np.asarray(vals)
    chosen_val = vals[np.random.randint(0, len(vals), n)]
    return chosen_val


def set_random_seed():
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))  # random seed for parallel processing


def error_task(model, decoder, ori_std, contrast, n_reps):
    # ori_std = stim_ori
    set_random_seed()  # random seed for parallel processing
    if decoder == 'CRB':
        crb_ave = model.cramer_rao(ori_std, n_reps=n_reps)
        crb_std = np.sqrt(crb_ave)
        summarydata = {'ori_std': ori_std, 'decoder': decoder, 'contrast': contrast,
                       'abs_error': None, 'std': crb_std}
    else:
        est_ori = model.decode_response(decoder, ori_std, contrast, n_reps)
        stdev = np.std(est_ori)
        abs_error = np.mean(abs(est_ori - ori_std))
        summarydata = {'ori_std': ori_std, 'decoder': decoder, 'contrast': contrast,
                       'abs_error': abs_error, 'std': stdev}
    return pd.DataFrame(summarydata, index=[''])


def dprime_task(model, decoder, ori_std, contrast, n_reps):
    # from Goris et al. (2009) citing Green & Swets (1966)
    # d' = difference between the mean response to the pedestal alone (blank) and the mean response to the signal (stimulus),
    #      and dividing this difference by the square root of the sum of their variances.
    set_random_seed()  # random seed for parallel processing
    est_ori = model.decode_response(decoder, ori_std, contrast, n_reps)
    abs_error = np.mean(abs(est_ori - ori_std))
    summarydata = {'ori_std': ori_std, 'decoder': decoder, 'contrast': contrast,
                   'abs_error': abs_error}
    return pd.DataFrame(summarydata, index=[''])


def median_discrim(model, decoder, ori_std, contrast, task_, **kwargs):
    # see Rokem & Silver (2009)
    # Build a distribution of the differences in orientation estimates to uncover the threshold (median)
    # Only works for ML and PV decoders, and for OriDis (not WTA or ConDet)
    set_random_seed()
    if task_ == 'ori_dis':
        est_1 = model.decode_response(decoder, ori_std, contrast, n_reps=kwargs.get('n_level_reps'))
        est_2 = model.decode_response(decoder, ori_std, contrast, n_reps=kwargs.get('n_level_reps'))
        # Calc distribution of differences
        diff = abs(est_1 - est_2)
    elif task_ == 'con_det':
        # Only works if you have access to the decoded estimate (otherwise can retrieve normalised threshold)
        # normalised threshold is meaningful for testing relative measures (e.g., OEI, or T across conditions)
        est_1 = model.detection(decoder, ori_std, stim_con=0, n_reps=kwargs.get('n_level_reps'))
        est_2 = model.detection(decoder, ori_std, stim_con=0, n_reps=kwargs.get('n_level_reps'))
        # # Calc distribution of differences
        # diff = abs(est_1 - est_2)
        # diff = normalise(diff, 0, 10)  # normalise differences to arbitrary scale
        diff = est_1 / est_2
        diff = np.exp(abs(np.log(est_1) - np.log(est_2)))
    else:
        return None
    # plt.hist(diff)
    median_diff = np.median(diff)
    stdev_diff = np.std(diff)
    summarydata = {'task': task_, 'ori_std': ori_std, 'decoder': decoder, 'contrast': contrast,
                   'threshold': median_diff, 'std': stdev_diff}
    summarydata = pd.DataFrame(summarydata, index=[''])
    return summarydata


def mocs_task(model, decoder, ori_std, contrast, task='', **kwargs):
    set_random_seed()  # random seed for parallel processing

    # if task not specified then give options
    if all(task != i for i in ['ori_id', 'ori_dis', 'con_det']):
        print("# # # TASK NOT SET # # #\nChoose from:\t'ori_id'\t,\t'ori_dis'\t,\t'con_det'")

    # load in externally saved file
    if model.neural_populations is None:
        model.neural_populations = model.get_attrib('neural_populations')

    levels_ = kwargs.get('levels')
    n_level_reps_ = kwargs.get('n_level_reps')

    # run method of constant stimuli
    rawdata = {'level': [], 'propn_corr': []}
    for i_level in levels_:  # loop through each level
        if any(i == task for i in ['ori_id', 'ori_dis']):  # for tasks where orientation is manipulated
            _split_rotation = False
            if _split_rotation is False:
                comp_ori = ori_std + i_level
                std_decoded = model.decode_response(decoder, ori_std, contrast, n_level_reps_)
                comp_decoded = model.decode_response(decoder, comp_ori, contrast, n_level_reps_)  # CLOCKWISE
                # DECISION
                if task == 'ori_dis':
                    is_corr = comp_decoded > std_decoded  # Proportion correct
                elif task == 'ori_id':
                    template_ori = model.gen_template_ori(ori_std, n_level_reps_)  # Generate template orientation
                    is_corr = abs(std_decoded - template_ori) < abs(comp_decoded - template_ori)  # Proportion correct
                else:
                    is_corr = None
            elif _split_rotation is True:
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
        if task == 'con_det':
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
        # print('\n', summarydata['decoder'], summarydata['ori_std'], summarydata['contrast'],
        #       summarydata['threshold'], sep='\n')
        return pd.DataFrame(summarydata, index=[''])
    else:
        summarydata = {'task': task, 'decoder': decoder, 'ori_std': ori_std, 'contrast': contrast,
                       'threshold': nan, 't_std': nan,
                       'slope': nan, 's_std': nan}
        print(f"no fit for Decoder={decoder} Ori={ori_std} Contrast={contrast}")
        return pd.DataFrame(summarydata, index=[''])


def false_function(model, decoder, ori_std, contrast, task='', **kwargs):
    print("I'm in")
    model.get_attrib('neural_populations')
    return 1


def staircase_task(model, decoder, ori_std, contrast, task, **kwargs):
    set_random_seed()  # Random seed for parallel processing
    # Init staircase
    info = kwargs['staircase_params']
    # Calculate all possible levels
    step_sizes = info['step_sizes']
    # Check that all steps a multiple of the smallest step
    if any((_i * 100) % (np.min(step_sizes) * 100) != 0 for _i in step_sizes):  # * 100 to avoid floating point errors
        print("Each step size must be a multiple of the smallest step")
    # Calculate all levels within bounds
    all_levels = [info['start_level']]
    loop_val = info['start_level']
    while loop_val < info['max_level']:
        loop_val = loop_val * 10 ** np.min(step_sizes)
        all_levels.append(loop_val)
    loop_val = info['start_level']
    while loop_val > info['min_level']:
        loop_val = loop_val / 10 ** np.min(step_sizes)
        all_levels.append(loop_val)
    all_levels = sorted(all_levels)

    # Time initial responses
    tic = time.time()
    # Decode multiple responses for standard and all possible comparisons (then loop through for trials)
    std_n_reps = info.get('std_n_reps', 5000)  # Larger number greatly determines reliability
    level_n_reps = info.get('level_n_reps', 500)  # More levels so can get away with smaller number
    if task == 'con_det':
        blank_resp = model.detection(decoder, ori_std, 0, n_reps=std_n_reps)
    if task == 'ori_dis':
        std_decoded = model.decode_response(decoder, ori_std, contrast, n_reps=std_n_reps)
    # Comparison responses for each level
    comparison = {}
    for i_level in all_levels:
        i_level = float(np.round(i_level, 3))  # Round to facilitate indexing (float for weird fix)
        if task == 'con_det':
            comparison[i_level] = model.detection(decoder, ori_std, stim_con=i_level, n_reps=level_n_reps)
        if task == 'ori_dis':
            comparison[i_level] = model.decode_response(decoder, ori_std + i_level, contrast, n_reps=level_n_reps)  # CW
            comparison[-i_level] = model.decode_response(decoder, ori_std - i_level, contrast, n_reps=level_n_reps)  # ACW
    comparison = {_k: list(comparison[_k]) for _k in comparison}  # to list
    # print(f"All responses: {time.time() - tic:.2f} sec")

    # Time staircase
    tic = time.time()
    # Init Staircase
    staircase = StaircaseHandler(start_level=info['start_level'], step_sizes=info['step_sizes'],
                                 n_up=info['n_up'], n_down=info['n_down'], n_reversals=info['n_reversals'],
                                 revs_per_thresh=info['revs_per_thresh'], extra_info=info.get('extra_info'),
                                 max_level=info['max_level'], min_level=info['min_level'])
    # Run staircase
    results = init_cols(['threshold', 'task', 'decoder', 'ori_std', 'contrast'])
    # Precompute random indices to select responses (much quicker than randomising on each loop)
    std_idcs_ = np.arange(0, std_n_reps)
    level_idcs_ = np.arange(0, level_n_reps)
    for _ in range(info.get('n_runs', 1)):
        # Precompute random indices to select responses (much quicker than randomising on each loop)
        std_idcs = shuffle(std_idcs_)
        level_idcs = shuffle(level_idcs_)
        # Init staircase
        staircase.continue_staircase = True
        while staircase.continue_staircase:
            i_level = np.round(staircase.current_level, 3)  # Rounded for use with dict
            if task == 'con_det':
                # Compare response for stimulus at level contrast, and for stimulus at 0 contrast (noise)
                i_blank_resp = blank_resp[std_idcs.pop()]
                i_target_resp = comparison[i_level][level_idcs.pop()]
                is_corr = i_target_resp > i_blank_resp
            if any(i == task for i in ['ori_id', 'ori_dis']):  # For tasks where orientation is manipulated
                i_level = i_level * random_choice([1, -1])  # Randomise rotation of comparison
                # Get orientation estimates
                i_std_decoded = std_decoded[std_idcs.pop()]
                i_comp_decoded = comparison[i_level][level_idcs.pop()]
                # DECISION
                if task == 'ori_dis':
                    if i_level > 0:
                        is_corr = i_comp_decoded > i_std_decoded  # Clockwise
                    else:
                        is_corr = i_comp_decoded < i_std_decoded  # Anticlockwise
                elif task == 'ori_id':
                    # Generate template orientation
                    template_ori_cw = model.gen_template_ori(ori_std)
                    # Clockwise
                    is_corr = abs(i_std_decoded - template_ori_cw) < abs(i_comp_decoded - template_ori_cw)
                else:
                    is_corr = None
            # Update staircase from model responses
            staircase.is_correct(is_corr)  # input if correct or incorrect and update staircase accordingly
            is_corr = None  # Reset
            # End of staircase
            if not staircase.continue_staircase:
                # Add run information
                staircase.add_run_info(task=task, decoder=decoder, ori_std=ori_std, contrast=contrast)  # Run info
                for _k in staircase.run_info.keys():
                    results[_k] = [*results[_k], *staircase.run_info[_k]]
                # Reset staircase
                staircase.reset()
    results = pd.DataFrame(results)

    # # # Print statements for debugging
    # print(f"Staircase at {info.get('n_runs', 1)} runs: {time.time() - tic:.2f} sec")
    # print(f"Mean (SD) = {np.mean(results['threshold']):.3f} ({np.std(results['threshold']):.3f})")

    return results


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

# def run(experiment, model, decoders=('WTA', 'PV', 'ML'), stdoris=(0, 45), contrasts=(20, 40), n_runs=1,
#         design='mocs', run_type_='loop', **kwargs):
#     # kwargs   mocs: levels, n_level_reps
#     # kwargs  stair: Staircase
#
#     dataset = None  # init
#
#     # perform a single run on a predefined condition
#     if run_type_ == 'single':
#         dataset = experiment(model=model, decoder='WTA', ori_std=45, contrast=100,
#                              **kwargs)
#
#     # loop through each condition combination
#     if run_type_ == 'loop':
#         tic = time.time()
#         loopruns_data = []
#         for i in range(n_runs):
#             for i_stdori in stdoris:
#                 for i_decoder in decoders:
#                     for i_contrast in contrasts:
#                         i_dataset = experiment(model=model, decoder=i_decoder, ori_std=i_stdori, contrast=i_contrast,
#                                                **kwargs)
#                         loopruns_data.append(i_dataset)
#         dataset = pd.concat([i for i in loopruns_data])
#         # print(f"\nLooping through {len(stdoris) * len(decoders) * len(contrasts)} conds for {n_runs} runs"
#         #       f"\n\t\t\t\t\t{time.time() - tic:.2f} seconds")
#
#     # perform all conditions using multiprocessing
#     if run_type_ == 'multi_proc':
#         tic = time.time()
#         # repeat() used for constant variables
#         mp_iters = {'model': repeat(model), 'decoder': [], 'ori_std': [], 'contrast': []}
#         # create lists of all conditions combinations for n_runs
#         # args
#         for i in range(n_runs):
#             for i_stdori in stdoris:
#                 for i_decoder in decoders:
#                     for i_contrast in contrasts:
#                         mp_iters['decoder'].append(i_decoder)
#                         mp_iters['ori_std'].append(i_stdori)
#                         mp_iters['contrast'].append(i_contrast)
#         mp_args = zip(mp_iters['model'], mp_iters['decoder'], mp_iters['ori_std'], mp_iters['contrast'])
#         # kwargs
#         for i_kwarg in kwargs:
#             mp_iters[i_kwarg] = repeat(kwargs[i_kwarg])  # add kwargs in order
#         mp_kwargs = repeat(kwargs)
#
#         # Multiprocessing - perform task on each condition (args) with experiment params (kwargs)
#         args_for_starmap = zip(repeat(experiment), mp_args, mp_kwargs)
#         multi_proc = True  # toggle to False if you want to perform .starmap() without parallel processing
#         if multi_proc is True:
#             try:
#                 # input zipped conds in same order as arguments in function (task)
#                 with mlp.get_context('spawn').Pool() as pool:
#                     multiproc_data = pool.starmap(run_function_and_unpack_args_kwargs, args_for_starmap)
#                     pool.close()
#                     pool.join()
#             except Exception as err:
#                 print('\n!!!')
#                 log = logging.getLogger(__name__)
#                 log.exception(err)
#                 print('!!!\n')
#         else:
#             multiproc_data = list(itertools.starmap(run_function_and_unpack_args_kwargs, args_for_starmap))  # loops through
#         print(f"\nMultiprocessing with {len(stdoris) * len(decoders) * len(contrasts)} conds for {n_runs} runs"
#               f"\n\t\t\t\t\t{time.time() - tic:.2f} seconds")
#         multiproc_data = pd.concat([i for i in multiproc_data])
#         dataset = multiproc_data.sort_values(by=['decoder', 'ori_std', 'contrast'])
#
#     # Add info about design
#     dataset['design'] = design
#
#     return dataset


class StaircaseHandler:
    def __init__(self, start_level, step_sizes, n_up, n_down, n_reversals, revs_per_thresh,
                 min_level=None, max_level=None, extra_info=None):
        self.all_levels = None
        self.calculate_steps = True
        self.revs_per_thresh = revs_per_thresh
        self.n_down = n_down
        self.n_up = n_up
        self.n_reversals = n_reversals
        self.is_corr = []  # tracks n of corr answers in a row
        self.start_level = start_level
        self.current_level = start_level
        self.max_level = max_level
        self.min_level = min_level
        if max_level is None:
            self.max_level = start_level
        if min_level is None:
            self.min_level = 0
        self.step_sizes = step_sizes
        self.stepsizes_idx = 0
        self.current_stepsize = step_sizes[0]
        self.level_list = [start_level]
        self.current_direction = 'down'
        self.reversals = []
        self.continue_staircase = True
        self.threshold = []
        self.run_info = {'threshold': []}  # appends during runs
        self.results = None
        self.extra_info = extra_info  # to be added externally when saving to pkl

    def add_run_info(self, **kwargs):
        # Add in any custom run information
        for i_key in kwargs:
            try:
                self.run_info[i_key]  # check if key already created
            except KeyError:
                self.run_info[i_key] = []  # if key not created then init as list
            # append list with run info
            self.run_info[i_key].append(kwargs[i_key])

    def reset(self):
        # Reset initial values to defaults
        self.is_corr = []  # tracks n of corr answers in a row
        self.current_level = self.start_level
        self.stepsizes_idx = 0
        self.current_stepsize = self.step_sizes[0]
        self.level_list = [self.current_level]
        self.current_direction = 'down'
        self.reversals = []
        self.run_info = {'threshold': []}  # appends during runs

    def is_correct(self, is_correct):
        self.is_corr.append(is_correct)  # 0=incorrect 1=correct
        self.update_staircase()

    def update_level(self, current_direction=None):
        # Update level
        level = self.current_level
        stepsize = self.current_stepsize
        if current_direction is None:
            current_direction = self.current_direction
        if current_direction == 'up':
            level *= 10 ** stepsize
            if level > self.max_level:
                level = self.current_level  # Don't go past minimum
        elif current_direction == 'down':
            level /= 10 ** stepsize
            if level < self.min_level:
                level = self.current_level  # Don't go past minimum
        self.current_level = level  # Update current level - round for consistency with dicts
        if len(self.step_sizes) - 1 == self.stepsizes_idx:  # Hold on last stepsize once reached
            pass
        else:
            self.stepsizes_idx += 1

    def calc_threshold(self):
        self.threshold = geomean(self.reversals[-self.revs_per_thresh:])  # Geomean with log step sizes
        return self.threshold

    def update_staircase(self):
        self.current_stepsize = self.step_sizes[self.stepsizes_idx]
        # check if staircase needs to step down
        if len(self.is_corr) >= self.n_down:  # only check for step up when possible
            if all(i == 1 for i in self.is_corr[-self.n_down:]):
                if self.current_direction == 'up':  # if change in direction of staircase
                    self.reversals.append(self.current_level)
                self.current_direction = 'down'
                self.update_level()

        # check if staircase needs to step up
        if len(self.is_corr) >= self.n_up:  # only check for step up when possible
            if all(i == 0 for i in self.is_corr[-self.n_up:]):  # checks if incorrect answer
                if self.current_direction == 'down':  # if change in direction of staircase
                    self.reversals.append(self.current_level)
                self.current_direction = 'up'
                self.update_level()

        if len(self.reversals) == self.n_reversals:  # if predetermined n reversals reached, end exp
            self.continue_staircase = False
            threshold = self.calc_threshold()
            self.run_info['threshold'].append(threshold)

        if self.continue_staircase:
            self.level_list.append(self.current_level)

    def info2dframe(self, run_info=None):
        # Convert run info to DataFrame
        if run_info is None:
            run_info = self.run_info
        results = pd.DataFrame(run_info)
        orders = dict(ori=results['ori_std'].unique(), iv=results['contrast'].unique())
        custom_sort = {}
        for i_categ in orders:
            for idx, cond in enumerate(orders[i_categ]):
                custom_sort[cond] = idx
        results = results.sort_values(by=['ori_std', 'contrast'], key=lambda x: x.map(custom_sort))
        self.results = results

        return results

    def assign_levels(self, levels):
        # If providing present levels for staircase
        self.all_levels = levels
        self.calculate_steps = False


def check_if_skip_discrimination(all_combos, current_idx, ignore_list=None):
    # don't repeat ori_dis if conditions have previously been completed and only template params changing
    # as this only affects identification judgements
    if ignore_list is None:
        ignore_list = []
    _skip = False
    # reformat cond combos to exclude irrelevant conditions
    dis_combos = []
    all_combos = deepcopy(all_combos)
    for c in all_combos:  # loop through all cond combos
        for _i in ignore_list:
            c[_i] = None  # set params in ignore list to None (not relevant to ori_dis)
            dis_combos.append(deepcopy(c))
    current_combo = dis_combos[current_idx]  # current conditions
    prev_combos = dis_combos[:current_idx]  # only check across completed conds
    if any(_check == current_combo for _check in prev_combos):
        _skip = True
    return _skip


def remove_baseline_repeats(input_cond_combos, template_params, ignore=None):
    # TEMPLATE CONDS #
    # Set template cond and ensure no repeats for template conds when baseline is True
    if ignore is None:
        ignore = []
    cond_combos = deepcopy(input_cond_combos)
    for idx, _ in enumerate(cond_combos):
        i_conds = deepcopy(cond_combos[idx])
        if i_conds['template_baseline'] is True:
            for i_template_param in template_params:
                if any(_i == i_template_param for _i in ignore):
                    continue
                i_conds[i_template_param] = None  # when template baseline is True, other template params = None
            cond_combos[idx] = i_conds
    cond_combos = remove_list_repeats(cond_combos)
    return cond_combos


