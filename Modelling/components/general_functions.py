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


def asarray(*args, **kwargs):
    # creates function that forces to float16
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


def save_attrib(attrib, label, path=None):
    # save method for externally saved attribs
    # must save filenames in this format f"{label}.npy"
    if path is None:
        path = os.path.join(os.getcwd(), 'components', 'attribs')
    with open(os.path.join(path, f"{label}.pkl"), "wb") as file:
        pickle.dump(attrib, file, protocol=pickle.HIGHEST_PROTOCOL)


def get_attrib(label, path=None):
    # getter method for externally saved attributes
    # must save filenames in this format f"{label}.npy"
    if path is None:
        path = os.path.join(os.getcwd(), 'components', 'attribs')
    tic = time.time()
    with open(os.path.join(path, f"{label}.pkl"), 'rb') as file:
        attrib = pickle.load(file)
    print(f'pickle load: {time.time() - tic}')
    return attrib


def get_boundaries(base_ori, bound_range, abs_max_ori):
    # multiple boundaries across circular orispace (allows for shifting prefs window)
    base_oris = np.arange(base_ori - abs_max_ori, base_ori + abs_max_ori + 1, 180)
    base_oris = base_oris[(base_oris >= -abs_max_ori) & (base_oris <= abs_max_ori)]  # only within tiling range
    boundaries = [[]] * len(base_oris)
    for idx, val in enumerate(base_oris):
        boundaries[idx] = [val - (bound_range / 2), val + (bound_range / 2)]
    return boundaries


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


def sort_by_standard(standard, *args):
    # sorts arrays with inds matching ascending order of standard
    standard = np.asarray(standard)
    inds = standard.argsort()  # gets indices of sorted standard
    sorted_arrays = [standard[inds[::1]]]  # create list with first element as sorted standard
    for i_arg in args:
        i_arg = np.asarray(i_arg)
        sorted_arrays.append(i_arg[inds[::1]])  # append arrays sorted to inds of standard
    # sorted in ascending numerical order [::1]
    return sorted_arrays


def rolling_window(vector, window):
    # create rolling window across vector
    m = len(vector) - len(window)
    windows_idxs = hankel(np.arange(0, m + 1), np.arange(m, len(vector)))
    windows_vals = vector[windows_idxs]
    return windows_vals, windows_idxs


def generate_gaussian(x, mu, sigma):
    # gaussian tunings for multiple prefs (mu) & single halfwidth (sigma)
    mu = np.asarray([mu]).transpose()  # format to broadcast together with x
    p1 = 1 / (sigma * np.sqrt(2 * np.pi))
    p2 = -0.5 * ((x - mu) / sigma) ** 2
    tunings = p1 * np.exp(p2)
    return tunings


def generate_cosine(x, n_cycles=1, inverse=False):
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


def angles_equal(x, y, units='deg'):
    # compare whether two angles are equivalent by converting to radians and using remainder of pi
    if units == 'deg':
        x, y = np.radians(x), np.radians(y)
    # rounding provides precision up to 5 dec places
    return np.round(x % np.pi, 6) == np.round(y % np.pi, 6)


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
        z = 0
        # for sanity check, make sure np.var(array + noise, axis=1) == mean_val when scalar=1
        # np.var(array + noise) / np.mean(array) = FanoFactor
        # mean_fano = np.mean(np.var(array + noise, axis=1) / np.mean(array, axis=1))  # should approx scalar
        z = 0
    return noise

