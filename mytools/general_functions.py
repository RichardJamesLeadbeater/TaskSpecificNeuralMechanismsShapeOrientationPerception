import numpy as np


def generate_gaussian(x, mu, sigma, inverse=False):
    # gaussian tunings for multiple prefs (mu) & single halfwidth (sigma)
    mu = np.asarray([mu]).transpose()  # format to broadcast together with x
    p1 = 1 / (sigma * np.sqrt(2 * np.pi))  # why pi?
    p2 = -0.5 * ((x - mu) / sigma) ** 2
    curve = p1 * np.exp(p2)
    if inverse is True:
        curve = normalise(curve, val_start=0, val_end=np.max(curve), inverse=True)
    return curve


def generate_cosine(x, inverse=False):
    # shift so equal vals either side of 0
    x = x - np.mean(x)
    # convert x axis to rad and take cosine
    y = np.cos(x * (np.pi / 180))
    y = y - np.min(y)  # min=0
    y = y / np.max(y)  # max=1
    return y


def normalise(x, val_start=None, val_end=None, scalar=None, inverse=False):
    # from 0 to 1
    x = np.asarray(x)
    x = x - np.min(x)  # min = 0
    if np.max(x) != 0:  # prevent ZeroDiv
        x = x / np.max(x)  # max = 1
    if inverse is True:
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

