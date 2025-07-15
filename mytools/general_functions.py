import numpy as np


def poissonlike_noise(arr, scalar=1):
    arr = np.asarray(arr)
    if any(isinstance(arr, i) for i in [float, int]):  # if single scalar then match to array shape
        noise = np.sqrt(arr) * np.random.normal(0, scalar, arr.shape)
    else:
        scalar = (np.ones(arr.T.shape) * scalar).T
        noise = np.sqrt(arr) * np.random.normal(0, scalar)
    return noise


def find_nearest(array, value):  # nearest value and index in array for each value
    if isinstance(value, int) or isinstance(value, float):
        nearest_idx = (np.abs(array - value)).argmin()
        nearest_value = array[nearest_idx]
    else:
        array = np.asarray(array)
        value = np.asarray(value)
        nearest_idx = []
        nearest_value = []
        for val in value:
            nearest_idx.append((np.abs(array - val)).argmin())
            nearest_value.append(array[nearest_idx[-1]])
    return np.asarray(nearest_value), np.asarray(nearest_idx)

