from matplotlib import pyplot as plt
import numpy as np
from .general_functions import find_nearest, normalise


def measure_fwhm(tuning, tiling):
    # input tuning 1d array, output print of halfwidth
    # get max_tile
    max_val = np.max(tuning)
    max_idx = np.argmax(tuning)
    # check that vals are symmetric
    if tuning[max_idx - 300] == tuning[max_idx + 300]:
        print('tuning curve is symmetric')
    else:
        print('tuning curve is NOT symmetric')
    # get idcs where val == 0.5*max_tile
    half_val, half_idx = find_nearest(tuning, max_val / 2)  # nearest val
    # get distance between idcs
    fwhm_idx_distance = abs(half_idx - max_idx) * 2
    print(f"measured fwhm = {abs(tiling[0] - tiling[fwhm_idx_distance])}")


def max_resp_in_centre_of_window(obj, _print=True):
    # check that prefs window has stim_ori at it's centre by checking the tuned response
    # resp_tuned = obj.resp_tuned[:, 0]  # index in as n_level_reps has been applied
    resp_tuned = obj.trial_resp_tuned[:, 0]  # 1 trial of tuned response
    max_idx = (len(resp_tuned) - 1) / 2  # idx of max response
    max_idx = [int(np.floor(max_idx)), int(np.ceil(max_idx))]  # deals with decimal values
    if all(i != resp_tuned.argmax() for i in max_idx):
        if not any(resp_tuned[i - 1] <= resp_tuned[i] <= resp_tuned[i + 1] for i in max_idx):
            if _print is True:
                print(f"\nCircular tuning may have failed:"
                      f"\n\t- Stim_ori {obj.trial_stim_ori} causes max tuned response at pref_idx {resp_tuned.argmax()}"
                      f" rather than centre_idx {int(np.round(len(obj.trial_prefs) / 2))}")
            # plt.plot(obj.trial_prefs, resp_tuned)
            return False
        else:
            return True
    else:
        return True


def noise_correlation_between_neurons(arr, og_arr):
    max_idx = np.argmax(np.sum(arr, 1))  # gives max responding neuron
    neighbour_idx = max_idx + 1
    # if neighbour to max has low activity (e.g. nrns sampling 22 and 158)
    if all(i < 0.1 for i in arr[max_idx + 1]):
        neighbour_idx = max_idx - 1
        if all(i < 0.1 for i in arr[neighbour_idx]):
            print('\n>>> Issue with .debugger.noise_correlation_between_neurons\n')

    arr_coeff = np.corrcoef(arr[max_idx], arr[neighbour_idx])[0, 1]
    og_coeff = np.corrcoef(og_arr[max_idx], arr[neighbour_idx])[0, 1]
    if any(arr_coeff < i for i in [0.05, og_coeff]):
        print(f"Small noise correlation\n"
              f"OG = {og_coeff}\n"
              f"RESP = {arr_coeff}")

    return arr_coeff


def weighted_resp_plots(n, w, xaxis=None):
    n = normalise(x=n[:, 0])
    w = normalise(x=w[:, 0])
    plt.subplots()
    if xaxis is None:
        xaxis = [i for i in range(len(n))]
    plt.plot(xaxis, n)
    plt.plot(xaxis, w)
    plt.title = 'weighted pooling'


def pref_spacing(mod_prefs, min_rate, max_rate):
    mid_idx = int(0.5 * len(mod_prefs))
    print(f"\n# # #"
          f"\ndebugger.pref_spacing():"
          f"\n\n\tcoded pref spacing: {min_rate}"
          f"\n\tinitial pref spacing: {mod_prefs[1] - mod_prefs[0]}"
          f"\n\n\tcoded max spacing: {max_rate}"
          f"\n\tmax spacing: {abs(mod_prefs[mid_idx] - mod_prefs[mid_idx + 1])}"
          f"\n# # #")



