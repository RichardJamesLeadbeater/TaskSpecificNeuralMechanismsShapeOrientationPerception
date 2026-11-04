import itertools
from itertools import repeat
import random
import numpy as np
from numpy import asarray as arr, newaxis as nax
from scipy.linalg import hankel
import pandas as pd
import time
import multiprocessing as mp
import os
import pickle
from scipy.optimize import curve_fit
from datetime import datetime
import matplotlib.pyplot as plt
from numpy import nan
from scipy import special


def convert_to_circular_space(vals, space_range):
    """Wrap values around/within defined circular space"""
    bound = space_range // 2
    return ((arr(vals) + bound) % space_range) - bound


def find_nearest(input_array, values):
    """Find nearest value and index in array for each element of values."""

    input_array = np.asarray(input_array)
    values = np.asarray(values)

    # Flatten values for vectorised nearest-neighbour calculation.
    original_shape = values.shape
    values_flat = values.ravel()

    diff = np.abs(input_array[None, :] - values_flat[:, None])
    nearest_idx = diff.argmin(axis=1)
    nearest_values = input_array[nearest_idx]

    # Restore the shape of the input.
    nearest_idx = nearest_idx.reshape(original_shape)
    nearest_values = nearest_values.reshape(original_shape)

    # Return scalar for scalar input.
    if values.ndim == 0:
        return nearest_values.item(), nearest_idx.item()

    return nearest_values, nearest_idx


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


def generate_wrapped_gaussian(x, mu, sigma, period=np.pi, wrap_range=3):
    # gaussian tunings for multiple prefs (mu) & single halfwidth (sigma)
    x = np.asarray(x)
    mu = np.asarray(mu).reshape(-1, 1)  # shape (n_neurons, 1) for broadcasting

    tunings = np.zeros((len(mu), len(x)))
    for k in range(-wrap_range, wrap_range + 1):
        # Shift by multiples of period
        shifted_x = x + k * period
        # Gaussian on shifted x
        p1 = 1 / (sigma * np.sqrt(2 * np.pi))
        p2 = -0.5 * ((shifted_x - mu) / sigma) ** 2
        tunings += p1 * np.exp(p2)

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

    # Swindale (1998): von Mises tuning function
    p1 = kappa.T * np.cos(2 * (x[:, nax] - prefs))

    # Calculate normalisation in log space to avoid overflow
    log_normalisation = np.log(2 * np.pi) + np.abs(kappa) + np.log(special.i0e(kappa))

    # Normalise in log space, then exponentiate
    log_tunings = p1.T - log_normalisation
    tunings = np.exp(log_tunings)

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


def poissonlike_noise(array, scale=1):
    array = arr(array)
    # Convert scalar to array if needed
    if np.isscalar(scale):
        scale_arr = np.sqrt(scale)  # scalar broadcasts automatically
    else:
        if array.ndim == 3:
            scale_arr = np.sqrt(scale)[:, None, None]  # broadcast over repeats & orientations
        elif array.ndim == 2:
            scale_arr = np.sqrt(scale)[:, None]

    # Generate noise
    noise = np.sqrt(array) * np.random.normal(0, scale_arr, size=array.shape)

    # for sanity check, make sure np.var(array + noise, axis=1) == mean_val when scalar=1
    # np.var(array + noise) / np.mean(array) = FanoFactor
    # mean_fano = np.mean(np.var(array + noise, axis=1) / np.mean(array, axis=1))  # should approx scalar

    return noise
