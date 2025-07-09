import math
import numpy as np
from numpy import asarray as arr
from psychopy import visual, core
from mytools.general_functions import normalise
import cairo
import matplotlib.pyplot as plt


def stim_window_array(win, stimuli):
    # Draw onto back buffer and remove before flipping screen
    [_s.draw() for _s in stimuli]
    core.wait(0.05)  # Wait a moment to ensure rendering is buffered (optional)

    # Capture what's currently in the back buffer
    win.getMovieFrame(buffer='back')
    img_array = np.asarray(win.movieFrames[-1])
    img_array = np.flipud(img_array)  # Buffered image is flipped upside down

    # Clear buffer to avoid accumulation
    win.movieFrames = []
    win.clearBuffer()

    return img_array


def draw_circle(diameter, thickness, color=(0, 0, 0), bg_color=(0.5, 0.5, 0.5)):
    # Define image size and create a surface
    image_size = diameter + 2 * thickness
    bg_size = int(image_size * 1.2)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, bg_size, bg_size)
    ctx = cairo.Context(surface)

    # Set the background colour (mid-grey by default)
    ctx.set_source_rgb(*bg_color)
    ctx.rectangle(0, 0, bg_size, bg_size)
    ctx.fill()

    # Enable best anti-aliasing
    ctx.set_antialias(cairo.ANTIALIAS_BEST)

    # Set up the circle parameters
    radius = diameter / 2
    centre_x = centre_y = bg_size / 2

    # Draw the circle
    ctx.set_line_width(thickness)
    ctx.arc(centre_x, centre_y, radius, 0, 2 * np.pi)
    ctx.set_source_rgb(color[0], color[1], color[2])  # Black circle
    ctx.stroke()

    # Convert Cairo surface to NumPy array for further processing if needed
    buf = surface.get_data()
    img_array = np.ndarray(shape=(bg_size, bg_size, 4), dtype=np.uint8, buffer=buf)

    return img_array


def stim_as_array(win, input_stim, size_pix=None, output_type='array'):  # input a visual.GratingStim stimulus
    """creates a matching stimulus with array stored in output.image"""
    # output_type either GratingStim or ImageStim ('grating', 'image')
    if size_pix is None:
        size_pix = win.size
    elif any(isinstance(size_pix, i_) for i_ in [int, float]):
        size_pix = [size_pix, size_pix]
    if not any(isinstance(input_stim, _t) for _t in [list, tuple, set]):
        input_stim = [input_stim]  # must be list for .BufferImageStim()
    # rect (norm units) selects area of screen to buffer RGB array from
    rect_w = (size_pix[0] / win.size[0])  # size[0] / width = norm_units width
    rect_h = (size_pix[1] / win.size[1])  # size[1] / height = norm_units height
    # buffers a stimulus defined from input stim
    win_color_og = win.color
    win.color = [0, 0, 0]  # removes any gcorr of bg
    buffer_stimulus = visual.BufferImageStim(
        win=win, stim=input_stim, rect=(-rect_w, rect_h, rect_w, -rect_h),
        buffer='back', pos=(0, 0),
        interpolate=True  # interpolate?  not changing anything from here
    )
    # Note: calling buffer.image after this point causes image to be drawn on screen (flash)
    win.color = win_color_og  # reset to original bg
    # extract array of RGB255 values
    rgb255_array = arr(buffer_stimulus.image)[:, :, 0]  # greyscale as all dims equal
    rgb_array = (rgb255_array - 127.5) / 127.5  # .ImageStim.ColorSpace = rgb
    # print(f"min={rgb_array.min()} max={rgb_array.max()}")
    rgb_array = rgb_array[:np.min(rgb_array.shape), :np.min(rgb_array.shape)]  # trim uneven parts of array
    rgb_array = np.fliplr(rgb_array)  # flip array to counteract fliplr caused by .BufferImageStim()
    if output_type == 'grating':
        output_stim = visual.GratingStim(
            win=win, tex=rgb_array, units='pix', size=size_pix,  # size in pixels
            blendmode=win.blendMode,
            interpolate=True,
        )
    elif output_type == 'image':
        output_stim = visual.ImageStim(
            win=win, image=rgb_array, units='pix',
            size=rgb_array.shape,  # same size as rgb_array (prevent any strange interpolation / resizing)
            opacity=1, interpolate=True,
        )
    else:
        output_stim = rgb_array
    win.clearBuffer()  # prevent buffer image being shown on screen upon win.flip()
    return output_stim


def make2DGauss(x, y, mean=0.0, sd=1.0, gain=1.0, base=0.0):
    # Lifted from PsychoPy
    """
    Return the gaussian distribution for a given set of x-vals

   :Parameters:
        x,y should be x and y indexes  as might be created by np.mgrid
        mean: float
            the centre of the distribution - may be a tuple
        sd: float
            the width of the distribution - may be a tuple
        gain: float
            the height of the distribution
        base: float
            an offset added to the result

    """

    if np.size(sd) == 1:
        sd = [sd, sd]
    if np.size(mean) == 1:
        mean = [mean, mean]

    simpleGauss = np.exp(
        (-np.power(x - mean[0], 2) / (2 * sd[0] ** 2)) - (np.power(y - mean[1], 2) / (2 * sd[1] ** 2)))
    return base + gain * (simpleGauss)


def apply_gaussian_mask(array, sd=3, rect_pix=None):
    # input: array of luminance values (-1 to 1)
    # output: array of luminance values with gaussian filter applied
    size_pix = array.shape[0]
    xy = np.mgrid[0:size_pix, 0:size_pix]
    mean_idx = int(np.round(0.5 * (size_pix - 1), 0))
    # sd_pix = size_pix * sd
    if rect_pix is None:
        sd_pix = size_pix / (sd * 2)
    else:
        sd_pix = rect_pix / (sd * 2)  # apply gaussian mask to a central portion of the array
    gauss_2d = make2DGauss(xy[0], xy[1], mean_idx, sd_pix)
    gauss_2d = normalise(gauss_2d, 0, 1)
    filtered_array = array * gauss_2d
    return filtered_array


def get_radial_pos_locs(centre_pos, shift, loc_labels=None):
    """
    Generate radial positions for the cardinal and diagonal points around the centre
    at a specified distance (shift). If loc_labels is provided, only the specified
    locations will be returned.

    :param centre_pos: List or tuple of [x, y] coordinates for the centre position.
    :param shift: Radial distance from the centre.
    :param loc_labels: Optional list of locations to return (if provided).
    :return: Dictionary of radial positions.
    """
    # Define the angles for each of the 8 cardinal and diagonal points
    angles = {loc: angle for loc, angle in zip(
        ['top', 'right', 'bottom', 'left', 'top_right', 'bottom_right', 'bottom_left', 'top_left'],
        [90, 0, 270, 180, 45, 315, 225, 135]
    )}

    # Generate positions based on polar coordinates (angle and radius)
    pos_locs = {}
    for loc, angle in angles.items():
        # Convert angle to radians and calculate x, y
        x = centre_pos[0] + shift * math.cos(math.radians(angle))
        y = centre_pos[1] + shift * math.sin(math.radians(angle))
        pos_locs[loc] = [x, y]

    # If loc_labels is provided, filter the positions to include only the selected ones
    if loc_labels is not None:
        pos_locs = {_k: pos_locs[_k] for _k in loc_labels}  # Index chosen keys

    return pos_locs


def get_central_coordinates(image):
    """
    Given an image (2D or 3D), returns the coordinates of the central pixel.
    Prints a warning if there is no true center (i.e., when dimensions are even).

    :param image: Input image, numpy array of shape (height, width) or (height, width, channels).
    :return: Tuple of central coordinates (row, col).
    """
    # Get the height and width of the image
    height, width = image.shape[:2]

    # Check if the height and width are even (indicating no true center)
    if height % 2 == 0 or width % 2 == 0:
        print("Warning: Image does not have a true center due to even dimensions.")

    # Calculate the central coordinates
    central_row = height // 2
    central_col = width // 2

    return central_row, central_col


def deg_to_px(size_deg=None, monitor=None, screen_width_px=None, screen_width_cm=None, viewing_distance_cm=None):
    """
    Convert visual angle in degrees to pixels on screen.
    """
    # Set settings from monitor if provided
    if monitor is not None:
        screen_width_px = monitor.getSizePix()[0]
        screen_width_cm = monitor.getWidth()
        viewing_distance_cm = monitor.getDistance()

    # Convert degrees to radians
    size_radians = np.deg2rad(size_deg)

    # Physical size on screen in cm (visual angle formula)
    size_cm = 2 * viewing_distance_cm * np.tan(size_radians / 2)

    # Calculate pixel density per cm
    px_per_cm = screen_width_px / screen_width_cm

    # Convert size in deg to pix
    size_pix = px_per_cm * size_cm

    # Round to nearest int
    size_pix = int(np.round(size_pix))

    return size_pix


def px_to_deg(size_pix=None, monitor=None, screen_width_px=None, screen_width_cm=None, viewing_distance_cm=None):
    """
    Convert size in pixels on screen to visual angle in degrees.
    """
    # Set settings from monitor if provided
    if monitor is not None:
        screen_width_px = monitor.getSizePix()[0]
        screen_width_cm = monitor.getWidth()
        viewing_distance_cm = monitor.getDistance()

    # Calculate cm per pixel
    cm_per_px = screen_width_cm / screen_width_px

    # Convert size from pixels to cm
    size_cm = size_pix * cm_per_px

    # Convert physical size to visual angle (in degrees)
    size_radians = 2 * np.arctan(size_cm / (2 * viewing_distance_cm))
    size_deg = np.rad2deg(size_radians)

    return size_deg


