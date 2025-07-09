import numpy as np
import os
import pandas as pd


def load_gamma_grid(calib_name):
    # grid_path = r"C:\Users\rjl54\Documents\Psychophysics\Checks and Tests\GammaCalibration\gamma_grids"
    grid_path = r"C:\Users\rjl54 Admin\PycharmProjects\Psychophysics\Checks and Tests\GammaCalibration\gamma_grids"
    gamma_grid = pd.read_csv(os.path.join(grid_path, calib_name))
    return gamma_grid


def gamma_lut(monitor, lum_stepsize=0.01):
    # sample luminance vals from 0 to 255 and apply gamma correction
    n_decimals = str(lum_stepsize)[::-1].find('.')  # n_decimals specifies the resolution of the gLUT
    desired_lums = np.arange(0, 255 + lum_stepsize, lum_stepsize)
    desired_lums = np.round(desired_lums, n_decimals)  # round to n decimals for dict keys
    desired_lums = np.tile(desired_lums, [3, 1])  # linearise each rgb gun independently
    linearised_lums = monitor.linearizeLums(desired_lums.T) * 255  # transpose necessary to work with PsychoPy function
    lookuptable = {}  # LUT will have keys as lums (0 to 255) and vals as list of 3 linearised lums (RGB)
    for idx_lum, lum in enumerate(desired_lums[0, :]):
        lookuptable[f'{lum}'] = linearised_lums[idx_lum]  # desiredLum val now a key with linearised RGB vals as its paired value
    return lookuptable


def linearise_image(input_image, g_lut):
    # linearise the output of each RGB gun independently using gammagrid
    # requires 2D or 3D greyscale image (all values equal on dim3)
    image = np.asarray(input_image)
    image = (image * 127.5) + 127.5
    n_decimals = str(list(g_lut)[1])[::-1].find('.')  # resolution of the gLUT
    image = np.round(image, n_decimals)  # match format of LUT keys
    linearised_image = None  # init
    if image.ndim == 1:  # i.e. one RGB triplet
        if all(image[0] == i_ for i_ in image):  # check all same coded lum vals
            linearised_image = g_lut[f"{image[0]}"]
        else:
            linearised_image = np.empty(shape=[len(image), 1, 3], dtype=image.dtype)  # init new RGB image
            for idx_, lum in enumerate(image):
                linearised_image[idx_, :] = g_lut[f"{lum}"]
    elif any(len(image.shape) == i_ for i_ in [2, 3]):  # if 2D or 3D image
        linearised_image = np.empty(shape=[image.shape[0], image.shape[1], 3], dtype=image.dtype)  # init new RGB image
        # loop through each dimension of image
        for idx1, lum_row in enumerate(image):  # loops in 1st dim (rows) of img (e.g. img[i, :, :] )
            for idx2, lum in enumerate(lum_row):  # loops in 2nd dim (cols) of img (e.g. img[i, j, :] )
                try:
                    linearised_image[idx1, idx2, :] = g_lut[f"{lum}"]  # pulls rgb triplet for each pixel = img[i, j, :]
                except:
                    z = 0
    else:
        return None
    linearised_image = (linearised_image - 127.5) / 127.5  # convert to -1to1
    return linearised_image

