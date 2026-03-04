import numpy as np
import random
import cv2
import moten
import matplotlib.pyplot as plt
from scipy.stats import zscore as zs

from classes.natspatpred import NatSpatPred
NSP = NatSpatPred()
NSP.initialise(verbose=False)

def isotropic_gaussian(dims:tuple, sigma:float):
    x = np.arange(0, dims[0], 1, float)
    y = np.arange(0, dims[1], 1, float)
    x, y = np.meshgrid(x, y)
    x0 = dims[0] // 2
    y0 = dims[1] // 2
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

def circle_stim(dims:tuple, radius:float, center:tuple=None):
    dot_img = np.zeros((dims[0], dims[1]))
    if center is None:
        center = (dims[0] // 2, dims[1] // 2)
    
    y, x = np.ogrid[-center[0]:dims[0]-center[0], -center[1]:dims[1]-center[1]]
    mask = x*x + y*y <= radius*radius

    dot_img[mask] = 1
    return dot_img

def random_dot(dims:tuple, n_dots:int, dot_rad:int):
    dot_img = np.zeros(dims)
    for i in range(n_dots):
        x = random.randint(dot_rad, dims[0]-dot_rad)
        y = random.randint(dot_rad, dims[1]-dot_rad)
        cv2.circle(dot_img, (y, x), dot_rad, (1,), -1)
    return dot_img

def show(input:np.ndarray, cmap:str='binary', figsize:tuple=(6,6), invert_y:bool=True):
    """
    Display an image using matplotlib.

    Parameters:
    input (np.ndarray): The input image.
    cmap (str): The color map to use for displaying the image. Default is 'binary'.
    figsize (tuple): The size of the figure. Default is (6,6).
    """
    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(input, cmap=cmap)
    ax.axis("off")
    
    if invert_y:
        ax.invert_yaxis()  # Invert the y-axis
    plt.show()

def show_random_filter(pyramid, filter):
    # The indices for the filters that are within the patch
    filter_indices = np.where(filter == True)[0]
    this_filter = random.choice(filter_indices)
    pyramid.view.show_filter(this_filter)
    plt.show()

def cut_corners(input:np.ndarray, cut_size:int):
    """
    Applies a corner-cutting filter to the input array.

    Args:
        input (np.ndarray): The input array.
        cut_size (int): The size of the corner to be cut.

    Returns:
        np.ndarray: The filtered array with corners cut.
    """
    bool_mask = np.ones_like(input, dtype=bool)
    bool_mask[cut_size:-cut_size, cut_size:-cut_size] = False

    return input * ~bool_mask

def make_checker(dims, checkercenter, scales, scaling_factor, checker_size, stride):
    """
    Create a checkerboard pattern with optional scaling and centering.

    Args:
        dims (tuple): The dimensions of the checkerboard.
        checkercenter (tuple): The center coordinates of the checkerboard.
        scales (int): The number of scales for the checkerboard.
        scaling_factor (float): The scaling factor for each scale.
        checker_size (int): The size of each checkerboard square.
        stride (int): The stride for the smaller checkerboard.

    Returns:
        numpy.ndarray: The generated checkerboard pattern.
    """
    
    if scales == 0:
        # Create a full checkerboard of the given dimensions and checker size
        checkerboard = np.indices(dims).sum(axis=0) % 2
        checkerboard = np.repeat(checkerboard, checker_size, axis=0)
        checkerboard = np.repeat(checkerboard, checker_size, axis=1)
        checkerboard = checkerboard[:dims[0], :dims[1]].astype(float)
        return checkerboard
    else:
        # Create a checkerboard of the current scale
        checkerboard = np.indices(dims).sum(axis=0) % 2
        checkerboard = np.repeat(checkerboard, checker_size, axis=0)
        checkerboard = np.repeat(checkerboard, checker_size, axis=1)
        checkerboard = checkerboard[:dims[0], :dims[1]].astype(float)

        # Create a smaller checkerboard in the center
        smaller_dims = (int((dims[0] - 2 * stride) / scaling_factor), int((dims[1] - 2 * stride) / scaling_factor))
        smaller_checker_size = int(checker_size / scaling_factor)
        smaller_checkerboard = make_checker(smaller_dims, checkercenter, scales - 1, scaling_factor, smaller_checker_size, stride)

        # Replace the section of the larger checkerboard with the smaller one
        start = (dims[0] // 2 - smaller_dims[0] // 2, dims[1] // 2 - smaller_dims[1] // 2)
        checkerboard[start[0] + stride:start[0] + smaller_dims[0] + stride, start[1] + stride:start[1] + smaller_dims[1] + stride] = smaller_checkerboard

        return checkerboard

def plot_filter_locations(gabor_pyramid, in_range, pixdims=(425, 425), pix_per_dim:float=(425/8.4), filter_list:list=None):
    """
    Plots the locations of filters in a Gabor pyramid.

    Parameters:
    - gabor_pyramid: The Gabor pyramid object.
    - in_range: A boolean array indicating whether each filter is within the desired range.
    - pixdims: The dimensions of the image in pixels. Default is (425, 425).
    - pix_per_dim: The number of pixels per dimension. Default is 425/8.4.

    Returns:
    None
    """
    
    # Initialize an empty array with shape (nfilters, 2)
    coordinates = np.empty((gabor_pyramid.view.nfilters, 2))

    # Fill in the values
    for filter in range(gabor_pyramid.view.nfilters):
        coordinates[filter, 0] = gabor_pyramid.view.filters[filter]["centerh"] * (
            pixdims[0] / pix_per_dim
        )
        coordinates[filter, 1] = gabor_pyramid.view.filters[filter]["centerv"] * (
            pixdims[1] / pix_per_dim
        )

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the locations of all filters, making the ones outside the range transparent
    # for i in range(gabor_pyramid.view.nfilters):
    for i in range(len(filter_list)):
        if in_range[i]:
            ax.plot(coordinates[i, 0], coordinates[i, 1], 'o', color='red', alpha=0.1)
        else:
            ax.plot(coordinates[i, 0], coordinates[i, 1], 'o', color='blue', alpha=0.1)

    # Set the limits of the plot to the size of the image
    ax.set_xlim(0, pixdims[0] / pix_per_dim)
    ax.set_ylim(0, pixdims[1] / pix_per_dim)

    # Show the plot
    plt.show()
    
def plot_filter_outputs(sel_output, filters_per_freq_sel, spat_freqs, img_indices=[0, 1]):
    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    start_idx = 0

    for i, ax in enumerate(axes.flat):
        for img in img_indices:
            ax.plot(sel_output[img][start_idx:start_idx+filters_per_freq_sel[i]])
        
        start_idx += filters_per_freq_sel[i]
        ax.set_title(f"Gabor filter pyramid output \nSpatial frequency {spat_freqs[i]/8.4} cycles per degree")

    plt.show()
  
def normalize_output(output, n_spatfreqs, filters_per_freq):
    """
    Normalizes the output of the Gabor pyramid.

    Parameters:
    - output: The output of the Gabor pyramid.

    Returns:
    - The normalized output.
    """
    start = 0
    output_norm = np.zeros_like(output)

    for i in range(n_spatfreqs):
        end = start + filters_per_freq[i]
        output_norm[:, start:end] = zs(output[:, start:end])
        start = end

    return output_norm

# ### This one loops over the different directions, and kind of woroks, but the outputs need to be concatenated, stacked manually
# def select_filters(
#     pyramid,
#     output,
#     imgs: np.ndarray,
#     img_no: int,
#     spat_freqs: list[float],
#     direction_masks: list|tuple,
#     filters_per_freq: list[float],
#     percentile_cutoff: float = 99,
#     plot: bool = False,
#     verbose: bool = False,
# ):
#     """
#     Selects filters from a pyramid based on the given parameters.
#     """
#     this_output=output[img_no]
#     n_spatfreqs = len(spat_freqs)

#     # Initialize the aggregated results
#     output_norm_agg = []
#     filter_selection_agg = []
#     filters_per_freq_sel_agg = []

#     # Loop over the different directions
#     for dir_no in range(len(direction_masks)):
#         # Normalize the output for every spatial frequency separately
#         start = 0
#         output_norm = []
#         for i in range(n_spatfreqs):
#             end = start + filters_per_freq[i]
#             this_dir_mask = direction_masks[dir_no][start:end]
#             output_norm.append(zs(this_output[start:end] * this_dir_mask))
#             start = end

#         # Calculate the nth percentile for each spatial frequency and create a boolean mask
#         filter_selection = []
#         filters_per_freq_sel = []
#         for i in range(n_spatfreqs):
#             percentile = np.percentile(output_norm[i], percentile_cutoff)
#             mask = output_norm[i] > percentile
#             filter_selection.append(mask)
#             n_filters = np.sum(mask)
#             filters_per_freq_sel.append(n_filters)
#             if verbose:
#                 print(
#                     f"Direction {dir_no}, Spatial frequency {i}: percentile = {percentile}, number of values > percentile = {n_filters}"
#                 )

#         # Aggregate the results
#         output_norm_agg.append(output_norm)
#         filter_selection_agg.append(filter_selection)
#         filters_per_freq_sel_agg.append(filters_per_freq_sel)

#     # Boolean mask to filter out the selected filters
#     filter_selection_agg = [np.concatenate(fs) for fs in filter_selection_agg]

#     # Dictionary of the selected filters
#     filter_selection_dictlist = [list(np.array(pyramid.view.filters)[fs]) for fs in filter_selection_agg]
    
#     if verbose:
#         for dir_no, fs in enumerate(filter_selection_agg):
#             print(
#                 f"Direction {dir_no}: Filter includes {np.sum(fs)} out of {pyramid.view.nfilters} filters"
#             )
#     if plot:
#         for dir_no, fs in enumerate(filter_selection_agg):
#             show(imgs[img_no], figsize=(6, 6))
#             plot_filter_locations(
#                 gabor_pyramid=pyramid, in_range=fs, pixdims=(425, 425)
#             )
        
#     return output_norm_agg, filters_per_freq_sel_agg, filter_selection_agg, filter_selection_dictlist


def select_filters(
    pyramid,
    filter_list,
    output,
    imgs: np.ndarray,
    img_no: int,
    spat_freqs: list[float],
    direction_masks: list|tuple,
    filters_per_freq: list[float],
    percentile_cutoff: float = 99,
    best_n: int = None,
    verbose: bool = False,
):
    """
    Selects filters from a pyramid based on the given parameters.
    """
    this_output=output[img_no]
    n_spatfreqs = len(spat_freqs)

    # Initialize the aggregated results
    output_norm_agg = []
    filter_selection_agg = []
    filters_per_freq_sel_agg = []

    # Loop over the different directions
    for dir_no in range(len(direction_masks)):
        # Normalize the output for every spatial frequency separately
        start = 0
        output_norm = []
        for i in range(n_spatfreqs):
            end = start + filters_per_freq[i]
            this_dir_mask = direction_masks[dir_no][start:end]
            output_norm.append(zs(this_output[start:end] * this_dir_mask))
            start = end

        # Calculate the nth percentile for each spatial frequency and create a boolean mask
        filter_selection = []
        filters_per_freq_sel = []
        for i in range(n_spatfreqs):
            if best_n is not None:
                # Select the top best_n filters
                sorted_indices = np.argsort(output_norm[i])
                mask = np.zeros_like(output_norm[i], dtype=bool)
                mask[sorted_indices[-best_n:]] = True
            else:
                # Use percentile cutoff
                percentile = np.percentile(output_norm[i], percentile_cutoff)
                mask = output_norm[i] > percentile

            filter_selection.append(mask)
            n_filters = np.sum(mask)
            filters_per_freq_sel.append(n_filters)
            if verbose:
                print(
                    f"Direction {dir_no}, Spatial frequency {i}: number of values > percentile = {n_filters}"
                )

        # Aggregate the results
        output_norm_agg.append(output_norm)
        filter_selection_agg.append(filter_selection)
        filters_per_freq_sel_agg.append(filters_per_freq_sel)

    # Boolean mask to filter out the selected filters
    filter_selection_agg = [np.concatenate(fs) for fs in filter_selection_agg]

    # Dictionary of the selected filters
    # filter_selection_dictlist = [list(np.array(pyramid.view.filters)[fs]) for fs in filter_selection_agg]
    filter_selection_dictlist = [list(np.array(filter_list)[fs]) for fs in filter_selection_agg]
    
    if verbose:
        for dir_no, fs in enumerate(filter_selection_agg):
            print(
                f"Direction {dir_no}: Filter includes {np.sum(fs)} out of {pyramid.view.nfilters} filters"
            )
    return output_norm_agg, filters_per_freq_sel_agg, filter_selection_agg, filter_selection_dictlist


def cut_paste_mask(
    masks: np.ndarray,
    peri_angle: int,
    peri_ecc: float,
    pix_per_deg: float = (425 / 8.4),
    verbose: bool = False,
    plot: bool = False,
) -> np.ndarray:
    """
    Cuts and pastes a patch from a given mask and places it at a specified location.

    Args:
        masks (np.ndarray): Array of masks. The first mask should be the gaussian mask, the second one
            should be the checkerboard mask. (or whatever mask you want to place)
        peri_angle (int): Polar angle of the desired patch location in degrees.
        peri_ecc (float): Eccentricity of the desired patch location in degrees.
        pix_per_deg (float, optional): Pixels per degree. Defaults to (425 / 8.4).
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        plot (bool, optional): Whether to plot the patch. Defaults to False.

    Returns:
        np.ndarray: The modified mask with the patch placed at the specified location.
    """
    
    # Convert angles from degrees to radians
    peri_angle_rad = np.deg2rad(peri_angle)
    # Convert polar coordinates to Cartesian coordinates
    peri_x = peri_ecc * np.cos(peri_angle_rad)
    peri_y = peri_ecc * np.sin(peri_angle_rad)
    peripheral_center = (peri_x, peri_y)

    # Convert Cartesian coordinates to pixel coordinates
    peripheral_center_pix = tuple(
        [int(pixloc * pix_per_deg) for pixloc in peripheral_center]
    )
    if verbose:
        print(
            f"The desired patch location has: {peri_ecc} degrees eccentricity at {peri_angle} degrees polar angle at coordinates {(round(peripheral_center[0],2), round(peripheral_center[1],2))}."
        )
        print(
            f"Which requires the patch center to be shifted {peripheral_center_pix[0]} pixels horizontally and {peripheral_center_pix[1]} pixels vertically."
        )

    gauss_area = np.array(masks[0] > 0.001).reshape(masks[0].shape)
    cut_ices = NSP.utils.get_bounding_box(gauss_area)

    check_patch = masks[1][
        cut_ices[0] : cut_ices[1], cut_ices[2] : cut_ices[3]
    ]

    peri_patch = np.zeros_like(masks[0])

    row_shift = peripheral_center_pix[1]
    col_shift = peripheral_center_pix[0]

    peri_patch[
        cut_ices[0] - row_shift : cut_ices[1] - row_shift,
        cut_ices[2] + col_shift : cut_ices[3] + col_shift,
    ] = check_patch
    
    if plot:
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(check_patch, cmap="gist_gray")
        axes[1].imshow(peri_patch, cmap="gist_gray")
        plt.tight_layout()
        
    return peri_patch

def location_based_selection(pyramid, bounds_prc:tuple, verbose:bool=True):
    """Function to select filters based on their location in the image demarcated
        by acquired bounds in image percentage.

    Args:
        pyramid (pyramid class): The input pyramid of gabor filters.
        bounds_prc (tuple): The bounds of the patch in percentage of the image.
        verbose (bool, optional): Print yes or no. Defaults to True.

    Returns:
        list: The list of filters encapsulated by the patch.
        list: A boolean vector indicating which filters are included.
    """    
    all_filt_dict = pyramid.view.filters

    # Convert list of DotDict objects to list of dictionaries
    all_filt_dict = [dict(d) for d in all_filt_dict]

    # Extract centerh and centerv values into separate lists
    centerh_values = [d['centerh'] for d in all_filt_dict]
    centerv_values = [d['centerv'] for d in all_filt_dict]

    # Create boolean mask for filters within bounds
    mask = ((np.array(centerh_values) > bounds_prc[0]) & 
            (np.array(centerh_values) < bounds_prc[1]) & 
            (np.array(centerv_values) > bounds_prc[2]) & 
            (np.array(centerv_values) < bounds_prc[3]))

    # Apply mask to list of dictionaries
    encapsulated_list = [d for d, m in zip(all_filt_dict, mask) if m]

    if verbose:
        print(f"Original number of filters: {len(all_filt_dict)}")
        print(f"Number of filters encapsulated by our patch: {len(encapsulated_list)}")
        
    return encapsulated_list, mask.tolist()

def filts_per_freq(pyr_pars:dict, filter_list:list):
    """Function to compute the number of filters per spatial frequency.

    Args:
        pyr_pars (dict): The pyramid parameter dictionary.
        filter_list (list): The list of filters in the (part of the) pyramid of interest.

    Returns:
        list: The number of filters per spatial frequency.
    """    
    # Figure out how many filters there are per spatial frequency
    filters_per_freq = []
    for sf in pyr_pars["spatial_frequencies"]:

        all_filters = filter_list
        count = sum(1 for d in all_filters if d.get("spatial_freq") == sf)
        filters_per_freq.append(count)

    return filters_per_freq

def orient_boolmask(filter_dictlist:list):
    """
    Function to create boolean masks for each unique orientation in the filter dictionary list.
    """
    # Get all directions
    directions = [
        filter_dictlist[i]["direction"] for i in range(len(filter_dictlist))
    ]

    # Get unique directions
    unique_directions = np.unique(directions)

    # Initialize an empty array to store the masks
    direction_masks = np.zeros(
        (len(unique_directions), len(filter_dictlist)), dtype=bool
    )

    # Create a mask for each unique direction
    for i, direction in enumerate(unique_directions):
        direction_masks[i] = np.array(directions) == direction

    return direction_masks

def gab_heatmap(filter_list, 
                filts_per_freq:list, 
                cmap:str="inferno", 
                dir_idx:int|str="all", 
                freq_idx:int|str="all", 
                single_filter:bool=False,
                verbose:bool=False,
                plot:bool=True):
    """Plotting function to inspect the filter selection using a heatmap

    Args:
        filter_list (list): List of dictionaries that contain the filter parameters
        filts_per_freq (list): The number of filters per spatial frequency
        cmap (str, optional): Colour map. Defaults to "inferno".
        dir_idx (int | str, optional): The directions to plot, can be an integer index or "all". Defaults to "all".
        freq_idx (int | str, optional): The spatial frequencies to include, can be an integer index or "all". Defaults to "all".
    """    
    filt_stack = []

    dirs = range(len(filter_list)) if dir_idx == "all" else [dir_idx]
    for dir in dirs:
        these_filts_per_freq = filts_per_freq[dir]
        dir_filts = range(len(filter_list[dir])) # The number of filter orientations, directions
        
        if freq_idx != "all":
            
            start_idx = 0 if freq_idx == 0 else sum(these_filts_per_freq[:freq_idx])
            end_idx = sum(these_filts_per_freq[:freq_idx+1]) if freq_idx < len(these_filts_per_freq) - 1 else len(dir_filts)
            
            dir_filts = dir_filts[start_idx:end_idx]
        
        
        if single_filter:
            dir_filts = [random.choice(dir_filts)]
        for filt_no in dir_filts:
            fdict = filter_list[dir][filt_no]
            this_gabor = moten.core.mk_3d_gabor(vhsize=(425,425), 
                                        centerh=fdict['centerh'],
                                        centerv=fdict['centerv'],
                                        direction=fdict['direction'],
                                        spatial_freq=fdict['spatial_freq'],
                                        spatial_env=fdict['spatial_env'],
                                        temporal_freq=fdict['temporal_freq'],
                                        filter_temporal_width=fdict['filter_temporal_width'],
                                        aspect_ratio=fdict['aspect_ratio'],
                                        stimulus_fps=fdict['stimulus_fps'],
                                        spatial_phase_offset=fdict['spatial_phase_offset'],)

            filt_stack.append(this_gabor[1])

    filt_stack = np.array(filt_stack)
    if verbose:
        print(filt_stack.shape)
    if plot:
        _ = plt.subplots(figsize=(8, 8))
        plt.imshow(np.mean(filt_stack, axis=0), cmap=cmap)
    
    return filt_stack