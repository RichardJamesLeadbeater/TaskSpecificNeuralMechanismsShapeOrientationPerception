import itertools
from itertools import repeat
import random
import numpy as np
from numpy import asarray as arr, newaxis as nax
from scipy.linalg import hankel
import pandas as pd
import time
import dill
import multiprocessing as mp
import os
import pickle
from scipy.optimize import curve_fit
from datetime import datetime
import matplotlib.pyplot as plt
from numpy import nan
from scipy import special


def find_nearest(array, value):  # nearest value and index in array for each value
    if isinstance(value, int) or isinstance(value, float):
        nearest_idx = (np.abs(array - value)).argmin()
        nearest_value = array[nearest_idx]
    else:
        array = np.asarray(array)
        value = np.asarray(value)
        nearest_idx = []
        nearest_value = []
        for val in np.atleast_1d(value):
            nearest_idx.append((np.abs(array - val)).argmin())
            nearest_value.append(array[nearest_idx[-1]])
    return np.asarray(nearest_value), np.asarray(nearest_idx)


def generate_cosine(x, n_cycles=1):
    # shift so equal vals either side of 0
    x = x - np.mean(x)
    # convert x axis to rad and take cosine
    x = np.radians(x)
    y = np.cos(x * n_cycles)
    if len(y) > 1:
        y = y - np.min(y)  # min=0
        y = y / np.max(y)  # max=1
    else:
        y = 1
    return y


def generate_vonmises(x, prefs, kappa, input_format='rad'):
    if input_format == 'deg':  # orispace in radians
        x = np.radians(x)
        prefs = np.radians(prefs)
    # see Swindale (1998) for clear equations and HWHM calculation
    p1 = kappa.T * (np.cos(2 * (x[:, nax] - prefs)))
    tunings = np.exp(p1).T
    normalisation_constant = 2 * np.pi * special.i0(kappa)
    tunings = tunings / normalisation_constant
    # plt.plot(tunings[::20, :].T)
    return tunings


def normalise(x, val_start=None, val_end=None, scalar=None, _invert=False):
    # from 0 to 1
    x = np.asarray(x)
    x = x - np.min(x)  # min = 0
    if np.max(x) != 0:  # prevent ZeroDiv
        x = x / np.max(x)  # max = 1
    if _invert is True:
        x = 1 - x  # min=1, max=0
    if all(i is None for i in [val_start, val_end]):
        return x

    # if either val==0 set to near-zero (ZeroDiv)
    if val_start == 0 or val_start is None:
        val_start = 1 * 10 ** -10
    if val_end == 0:
        val_end = 1 * 10 ** -10

    # transform to new space from val_start to val_end
    ratio = val_start / val_end
    y = (x * (1 - ratio) + ratio)  # scaled from 1 to ratio val
    if scalar is None:
        scalar = val_end  # scaled from val_start to val_end
    y = y * scalar  # transform to scaled units in new space
    return y


def run_function_and_unpack_args_kwargs(fn, args, kwargs):
    # performs function with packed args and kwargs
    return fn(*args, **kwargs)


def join_dicts(dict1, dict2):
    # only works with identical keys
    output = {}
    for key in dict1:  # init list in each key
        output[key] = []
    for key in dict1:  # join values from shared keys in each dict
        for idx, val_ in enumerate(dict1[key]):
            try:
                output[key].append([dict1[key][idx], dict2[key][idx]])
            except KeyError:
                print('\n>\terror when joining dicts\n')
    return output


def remove_list_repeats(listy):
    # Remove repeated items within a list
    listy = listy.copy()
    output_listy = []
    for idx, i_val in enumerate(listy):
        if not any(_j == i_val for _j in output_listy):
            output_listy.append(i_val)
    return output_listy


def get_condcombos(dicty, fixed_ivs=None, interaction_ivs=None, _save=False, savepath=None):
    # fixed_ivs determine which ones to vary
    # interaction_ivs determine which ones to mix together
    og_dicty = dicty.copy()
    if all(z is None for z in [fixed_ivs, interaction_ivs]):
        condcombo_list = listallcombos(og_dicty, _save, savepath)
    else:
        if interaction_ivs is None:
            interaction_ivs = []
        condcombo_list = []
        for f_iv in fixed_ivs:
            i_dicty = {}
            for key in og_dicty:
                if key == f_iv or any(key == _i for _i in interaction_ivs):  # for this fixed_iv
                    i_dicty[key] = og_dicty[key]  # vary singular params
                else:
                    i_dicty[key] = [og_dicty[key][0]]  # use first val pair (keep constant)
            condcombo_list += listallcombos(i_dicty, _save, savepath)
    condcombo_list = remove_list_repeats(condcombo_list)  # only unique vals in list
    return condcombo_list


def listallcombos(dicty, _save=False, savepath=None):
    # output list of condition combo dictys
    dicty = dicty.copy()
    # convert dicty to list of each cond vals
    cond_list = [dicty[i] for i in dicty]
    cond_keys = [i for i in dicty]  # keys will be used to reformat to dict later
    # get every possible combination of condition vals
    combos = list(itertools.product(*cond_list))
    # transform to list of dicts with original labels
    combo_list = [{} for _ in range(len(combos))]  # same len as combo list (allows for easy indexing with keys)
    for idx_, i_combo in enumerate(combos):  # loop cond combos
        for jdx_, j_cond in enumerate(i_combo):  # loop each cond
            cond_key = cond_keys[jdx_]
            combo_list[idx_][cond_key] = j_cond  # index into dict for this combo with approprite cond info

    if _save is True:
        pd.DataFrame(combo_list).to_csv(os.path.join(savepath, 'cond_list.csv'))

    return combo_list


def poissonlike_noise(array, scalar=1):
    array = arr(array)
    if any(isinstance(array, i) for i in [float, int]):  # if single scalar then match to array shape
        noise = np.sqrt(array) * np.random.normal(0, scalar, array.shape)
    else:
        scalar = np.sqrt(scalar)  # convert to Fano Factor (e.g., noise_scale of 2 results in Fano of 2)
        scalar_arr = (np.ones(array.T.shape) * scalar).T
        noise = np.sqrt(array) * np.random.normal(0, scalar_arr)

        # for sanity check, make sure np.var(array + noise, axis=1) == mean_val when scalar=1
        # np.var(array + noise) / np.mean(array) = FanoFactor
        # mean_fano = np.mean(np.var(array + noise, axis=1) / np.mean(array, axis=1))  # should approx scalar

    return noise

