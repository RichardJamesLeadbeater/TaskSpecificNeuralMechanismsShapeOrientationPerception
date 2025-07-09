import numpy as np
from skimage.transform import resize
from scipy.ndimage import gaussian_filter


def center_image_on_background(img_arr, bg_shape, bg_color=128):
    """
    Center an image onto a background array.

    background_shape: Tuple of (height, width) for the background array.
    image_array: The 2D image array to be centered.

    Returns:
    background: A new array with the image centered on it.
    """
    bg_height, bg_width = bg_shape
    img_height, img_width = img_arr.shape

    # Create the background array initialized to zeros
    background = np.ones((bg_height, bg_width), dtype=img_arr.dtype) * bg_color

    # Calculate the starting coordinates to center the image
    start_row = (bg_height - img_height) // 2
    start_col = (bg_width - img_width) // 2

    # Place the image in the center of the background
    background[start_row:start_row + img_height, start_col:start_col + img_width] = img_arr

    return background


def gen_gaussian_mask(shape, center, sigma):
    """
    Create a 2D Gaussian mask.
    shape: Tuple of (height, width) for the mask size.
    center: Tuple of (center_row, center_col) for the Gaussian center.
    sigma: Standard deviation of the Gaussian (controls spread).
    """
    rows, cols = shape
    y, x = np.meshgrid(np.arange(cols), np.arange(rows))
    gaussian_mask = np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma ** 2))
    return gaussian_mask


def get_pos_locs(centre_pos, out_pos, loc_labels=None):
    # Assumes square arrangement
    # c = constant if getting idx of image on top of image
    pos_locs = {
        'centre': [centre_pos, centre_pos],
        'top': [centre_pos, out_pos],
        'bottom': [centre_pos, -out_pos],
        'left': [-out_pos, centre_pos],
        'right': [out_pos, centre_pos],
        'top_left': [-out_pos, out_pos],
        'top_right': [out_pos, out_pos],
        'bottom_left': [-out_pos, -out_pos],
        'bottom_right': [out_pos, -out_pos]
    }
    if loc_labels is not None:
        pos_locs = {_k: pos_locs[_k] for _k in loc_labels}  # Index chosen keys

    return pos_locs


def get_img_locs(img_len, c=0, margins=True):
    # Assumes square image
    # c = constant if getting idx of image on top of image
    centre_idx = int(img_len / 2)
    if margins is True:
        near_idx = int(img_len / 4) + c
        far_idx = int(img_len / 4 * 3) - c
    else:
        near_idx = 0 + c
        far_idx = img_len - c
    img_locs = {
        'centre': [centre_idx, centre_idx],
        'top': [centre_idx, near_idx],
        'bottom': [centre_idx, far_idx],
        'left': [near_idx, centre_idx],
        'right': [far_idx, centre_idx],
        'top_left': [near_idx, near_idx],
        'top_right': [far_idx, near_idx],
        'bottom_left': [near_idx, far_idx],
        'bottom_right': [far_idx, far_idx]
    }

    return img_locs


def apply_multisampling_antialiasing(img_array, samples=4, cval=0):
    # Apply multisampling antialiasing to an image array.
    # Determine the high-resolution size
    high_res_shape = (img_array.shape[0] * samples, img_array.shape[1] * samples)
    # Upsample the array to the high-resolution grid
    high_res_array = resize(img_array, high_res_shape, anti_aliasing=False, order=0, preserve_range=True,
                            cval=cval)
    # # Smooth edges before downsampling
    # high_res_array = gaussian_filter(high_res_array, sigma=1 * samples)

    # Downsample back to the original resolution with antialiasing
    antialiased_array = resize(high_res_array, img_array.shape, anti_aliasing=True, preserve_range=True,
                               cval=cval)

    return antialiased_array


def rgb_conversion(vals, conversion_label='101_to_255'):
    # Convert between different formats of rgb
    vals = np.atleast_1d(vals)
    if conversion_label == '101_to_255':
        vals = (vals + 1) / 2 * 255
    elif conversion_label == '255_to_101':
        vals = vals / 255 * 2 - 1

    return vals


def rgb255_to_minus101(img_array):
    """Convert array 0 to 255 into -1 to 1"""
    output_img = (img_array / 127.5) - 1

    return output_img


def rgb_to_greyscale(image):
    """
    Convert RGB(A) image(s) to greyscale.
    Supports 3D (H, W, 3/4) or 4D (N, H, W, 3/4) input.

    Parameters:
        image (np.ndarray): RGB or RGBA image(s). Can be a single image or a batch.

    Returns:
        np.ndarray: Greyscale image(s) with shape (H, W) or (N, H, W).
    """
    # Ensure input is either a single image or a batch of images with 3 or 4 channels
    if image.ndim not in (3, 4):
        raise ValueError("Unsupported image shape")

    # Convert to greyscale by applying luminance-preserving weights to RGB channels
    # image[..., :3] selects the R, G, B channels (ignoring alpha if present)
    # The dot product computes: 0.2989*R + 0.5870*G + 0.1140*B for each pixel
    image_2D = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    return image_2D


def convert_coords_to_index_from_center(image_shape, coords):
    """
    Convert the given coordinates relative to the centre of the image to image indices.

    :param image_shape: Tuple (height, width) of the image.
    :param coords: Tuple (y, x) of the coordinates (relative to the centre [0, 0]).
    :return: Tuple (row, col) of image indices (integer values).
    """
    height, width = image_shape  # Image dimensions

    # Find the centre of the image (index position)
    centre_y, centre_x = height // 2, width // 2

    # Calculate the corresponding indices by adding the relative coordinates to the centre
    y, x = coords  # Relative to the centre
    row_idx = int(centre_y + y)  # Add the y-coordinate to the centre row
    col_idx = int(centre_x + x)  # Add the x-coordinate to the centre column

    # Ensure indices are within bounds
    row_idx = max(0, min(height - 1, row_idx))  # Ensure y is within [0, height-1]
    col_idx = max(0, min(width - 1, col_idx))  # Ensure x is within [0, width-1]

    return row_idx, col_idx

