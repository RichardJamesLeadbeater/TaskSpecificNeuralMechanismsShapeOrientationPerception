import copy
import numpy as np
from scipy.fftpack import fft2, ifft2
import cv2
from mytools.images import rgb_to_greyscale


def apply_sf_distortion(img, sf, scale, mode='bandpass', bandwidth=1.0, d_mask=None):
    """
    Apply spatial frequency distortion to an image using specified filter type.

    Parameters:
        img: Input image (2D array)
        sf: Central spatial frequency (cycles per pixel)
        scale: Displacement strength
        mode: 'lowpass', 'highpass', or 'bandpass'
        bandwidth: Bandwidth in octaves (for bandpass)
        d_mask: Optional distortion mask (which pixels to distort and relative distortion strength)
    """
    if len(img.shape) > 2:
        img = rgb_to_greyscale(img)  # Convert to greyscale using weighted sum of channels
    height, width = img.shape
    freq_filter = create_frequency_filter((height, width), sf, mode=mode, bandwidth=bandwidth)

    # test = False
    # if test is True:
    #     # Distortions of SFs in actual image (stronger for SFs near peak) - NOTE: doesn't simulate low SF distortion
    #     filtered_img = apply_sf_filter(img, sf, mode, bandwidth)
    #     filtered_img = filtered_img - np.min(filtered_img)  # min = 0
    #     filtered_img = filtered_img / np.max(filtered_img)  # max = 1
    #     distortion_h = filtered_img * np.random.choice([-1, 1], size=filtered_img.shape) * scale
    #     distortion_v = filtered_img * np.random.choice([-1, 1], size=filtered_img.shape) * scale
    #     distorted_image = remap_image(img, distortion_h, distortion_v)
    #     return distorted_image

    # Generate gaussian noise (std=1) to define pixel displacements in vertical and horizontal directions
    noise_h = np.random.randn(height, width)
    noise_v = np.random.randn(height, width)

    # Apply spatial frequency filter to noise to define pixel displacements at the desired frequency
    filtered_noise_h = apply_filter(noise_h, freq_filter)
    filtered_noise_v = apply_filter(noise_v, freq_filter)

    # Scale filtered noise to define magnitude of pixel displacements at that SF (i.e. size of distortion in pixels)
    filtered_noise_h = normalise_gaussian(filtered_noise_h)
    filtered_noise_v = normalise_gaussian(filtered_noise_v)

    # Rescale to desired mean absolute displacement (so that mean_abs_displacement = scale)
    filtered_noise_h = filtered_noise_h * (scale / np.mean(np.abs(filtered_noise_h)))
    filtered_noise_v = filtered_noise_v * (scale / np.mean(np.abs(filtered_noise_h)))

    # Remap image by applying pixel displacements in H and V directions and scaling mask (optional)
    distorted_img = remap_image(img, filtered_noise_h, filtered_noise_v, d_mask)

    return distorted_img


def create_frequency_filter(size, omega_peak, mode='bandpass', bandwidth=1.0, slope=20):
    """
    Generate a low-pass, high-pass, or band-pass filter in the spatial frequency domain.

    Parameters:
        size: Tuple (height, width) of the filter
        omega_peak: For bandpass: centre frequency (cpp);
                    For lowpass/highpass: cutoff frequency (cpp)
        mode: 'lowpass', 'highpass', or 'bandpass'
        bandwidth: Bandwidth in octaves for bandpass (ignored for others)
        slope: How smoothly the cutoff frequency is defined (higher value = sharper cutoff)

    Returns:
        2D frequency filter array.
    """
    height, width = size  # Unpack the dimensions of the image/filter

    u = np.fft.fftfreq(height)  # Generate frequency bins (cycles per pixel) for rows (vertical axis)
    v = np.fft.fftfreq(width)   # Generate frequency bins for columns (horizontal axis)

    # Create 2D frequency grids: U varies along columns (x freq), V along rows (y freq)
    # indexing='ij' ensures matrix-style indexing (row, col) rather than Cartesian (x, y)
    U, V = np.meshgrid(u, v, indexing='ij')

    # Compute radial spatial frequency (magnitude of frequency vector) at each point in the frequency domain
    omega = np.sqrt(U ** 2 + V ** 2)

    # Avoid divide-by-zero
    omega[omega == 0] = 1e-10

    if mode == 'lowpass':
        # Smooth transition low-pass using logistic (sigmoid)
        filt = 1 / (1 + np.exp((omega - omega_peak) * slope))

    elif mode == 'highpass':
        # Smooth transition high-pass
        filt = 1 / (1 + np.exp(-(omega - omega_peak) * slope))

    elif mode == 'bandpass':
        # Log-exponential band-pass filter (Wiecek et al. style)
        log_ratio = np.abs(np.log(omega / omega_peak))
        filt = np.exp(-(log_ratio ** 3 * np.log(2)) / (bandwidth * np.log(2)) ** 3)

    else:
        raise ValueError("Invalid mode. Choose from 'lowpass', 'highpass', or 'bandpass'.")

    return filt


def apply_filter(img, freq_filter):
    """
    Apply a frequency domain filter to a 2D signal.
    """
    return np.real(ifft2(fft2(img) * freq_filter))


def normalise_gaussian(arr):
    """
    Normalise an array to zero mean and unit standard deviation.
    """
    return (arr - np.mean(arr)) / np.std(arr)


def remap_image(image, displacements_h, displacements_v, d_mask=None):
    """
    Remap an image using horizontal and vertical displacement maps.

    Parameters:
        image (ndarray): Input 2D image (grayscale).
        displacements_h (ndarray): Horizontal displacement map (same shape as image).
        displacements_v (ndarray): Vertical displacement map (same shape as image).
        d_mask (ndarray, optional): Binary mask indicating where displacements should be applied.
                                    Defaults to all ones (i.e., apply everywhere).
                                    (same shape as as image)

    Returns:
        remapped_image (ndarray): The remapped image using the given displacements.
    """
    # If no mask is provided, apply displacements to the entire image
    if d_mask is None:
        d_mask = np.ones_like(image)

    # Get image dimensions
    h, w = image.shape

    # Create meshgrid of original x and y coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Apply displacements scaled by the mask (map_x & map_y defines coordinate of each pixel on x or y axis)
    map_x = np.float32(x + displacements_h * d_mask)
    map_y = np.float32(y + displacements_v * d_mask)

    # Remap the image using the computed coordinate maps
    # Nearest-neighbour interpolation and border reflection are used
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)


def apply_sf_filter(input_img, sf, mode='bandpass', bandwidth=1.0):
    """
    Apply a spatial frequency filter (lowpass, highpass, or bandpass) to an image.

    Parameters:
        input_img: 2D input image (e.g. greyscale).
        sf: Cutoff or peak spatial frequency (in cycles per pixel).
        mode: 'lowpass', 'highpass', or 'bandpass'.
        bandwidth: Bandwidth in octaves (used for bandpass only).

    Returns:
        Filtered image (2D array, real values).
    """
    # Copy original input_img (useful if debugging)
    img = copy.deepcopy(input_img)

    # Temporarily remove the mean luminance value (DC component)
    dc_component = np.mean(img)
    img = img - dc_component

    # Create SF filter and apply to image array
    freq_filter = create_frequency_filter(img.shape[:2], omega_peak=sf, mode=mode, bandwidth=bandwidth)

    # Apply a frequency domain filter to a 2D signal
    filtered_img = apply_filter(img, freq_filter)

    # Add mean value back to restore overall brightness
    filtered_img = filtered_img + dc_component

    return filtered_img


def normalise_image(img, min_val=0, max_val=255):
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min) * (max_val - min_val) + min_val


# import copy
# import numpy as np
# from scipy.fftpack import fft2, ifft2, fftshift
# import matplotlib.pyplot as plt
# import os
# import cv2
#
#
# def apply_sf_distortion(input_img, peak_sf, scale, d_mask=None):
#     # Isolate frequency band centred on the peak frequency - Distort it
#     # SF units given in cycles per pixel (so max SF = 0.5 because 2 pix/cyc)
#     # Bandwidth fixed at 1 in Wiecek et al. (2014)
#     img = copy.deepcopy(input_img)
#     # Settings
#     width = img.shape[0]
#     height = img.shape[1]
#     filter_bp = log_exponential_filter((height, width), omega_peak=peak_sf, b=1)
#     # Generate random noise with stdev of 1
#     noise_h = np.random.randn(height, width)  # Horizontal displacement noise
#     noise_v = np.random.randn(height, width)  # Vertical displacement noise
#     # Apply the band-pass filter to both horizontal and vertical noise
#     filtered_noise_h = apply_filter(noise_h, filter_bp)
#     filtered_noise_v = apply_filter(noise_v, filter_bp)
#     # Normalise noise such that stdev=1, mean should = 0
#     filtered_noise_h = (filtered_noise_h - np.mean(filtered_noise_h)) / np.std(filtered_noise_h)
#     filtered_noise_v = (filtered_noise_v - np.mean(filtered_noise_v)) / np.std(filtered_noise_v)
#     # Wiecek - 4, 6, 8, 10, 14 for certain SFs (but smaller img)
#     filtered_noise_h = filtered_noise_h * scale
#     filtered_noise_v = filtered_noise_v * scale
#     # Remap pixels in H and V direction
#     distorted_img = remap_image(img, filtered_noise_h, filtered_noise_v, d_mask)
#
#     return distorted_img
#
#
# def log_exponential_filter(size, omega_peak, b=1.0):
#     """
#     Generate a log-exponential filter (filters frequencies about the peak SF)
#     size: (height, width) of the filter.
#     omega_peak: Peak spatial frequency (cycles per pixel)
#     b: bandwidth in octaves.
#     """
#     # Radial SF frequency at each point in frequency domain
#     height, width = size
#     u = np.fft.fftfreq(height)  # Frequency values in y direction
#     v = np.fft.fftfreq(width)  # Frequency values in x direction
#     # todo define step
#     U, V = np.meshgrid(u, v, indexing='ij')
#     omega = np.sqrt(U ** 2 + V ** 2)  # sqrt because combined xy vectors
#     # Apply the log-exponential band-pass filter
#     filter_band = np.exp(-(np.abs(np.log(omega / omega_peak)) ** 3 * np.log(2)) / (b * np.log(2)) ** 3)
#
#     return filter_band
#
#
# def apply_filter(noise, filter_bp):
#     """
#     Apply a filter to noise in the frequency domain.
#     """
#     # Fourier transform the noise
#     noise_ft = fft2(noise)
#     # Apply the band-pass filter
#     filtered_noise_ft = noise_ft * filter_bp  # Ensure that xy of freq space in same directions
#     # Inverse Fourier transform to get filtered noise in the spatial domain
#     filtered_noise = np.real(ifft2(filtered_noise_ft))
#
#     return filtered_noise
#
#
# def remap_image(image, displacements_h, displacements_v, d_mask=None):
#     """
#     Remap an image based on the horizontal and vertical displacements.
#     """
#     if d_mask is None:
#         d_mask = np.ones(shape=image.shape)
#     width = image.shape[0]
#     height = image.shape[1]
#     # Create a meshgrid for pixel coordinates
#     x, y = np.meshgrid(np.arange(width), np.arange(height))
#     # Apply displacements
#     map_x = np.float32(x + displacements_h * d_mask)
#     map_y = np.float32(y + displacements_v * d_mask)
#     # Remap the image using the displacements
#     remapped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
#
#     return remapped_image
