import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

class Utilities():

    def __init__(self):
        pass
        
    # Utility function to visualize dictionary structures
    def print_dict_structure(self, d, indent=0):
        for key, value in d.items():
            print(' ' * indent + str(key))
            if isinstance(value, dict):
                self.print_dict_structure(value, indent + 4)
                
                
    def print_large(self, item):
        with np.printoptions(threshold=np.inf):
            print(item)
            
            
    def get_zscore(self, data, print_ars = 'y'):
        mean_value = np.mean(data)
        std_dev = np.std(data)

        # Calculate z-scores
        z_scores = (data - mean_value) / std_dev

        if print_ars == 'y':
            print("Original array:", data)
            print("Z-scores:", z_scores)
            
        return z_scores

    def cap_values(self, array = None, lower_threshold = None, upper_threshold = None):
        
        if upper_threshold is None:
            upper_threshold = np.max(array)
        else:
            # Identify values above the upper threshold
            above_upper_threshold = array > upper_threshold
            
            # Identify the highest value below the upper threshold
            highest_below_upper_threshold = array[array <= upper_threshold].max()

            # Replace values above the upper threshold with the highest value below the upper threshold
            array[above_upper_threshold] = highest_below_upper_threshold

        if lower_threshold is None:
            lower_threshold = np.min(array)
        else:
            # Identify values below the lower threshold
            below_lower_threshold = array < lower_threshold

            # Identify the lowest value above the lower threshold
            lowest_above_lower_threshold = array[array >= lower_threshold].min()

            # Replace values below the lower threshold with the lowest value above the lower threshold
            array[below_lower_threshold] = lowest_above_lower_threshold

        return array

    def mean_center(self, data, print_ars = 'y'):
        mean_value = np.mean(data)

        # Mean centering
        centered_data = data - mean_value

        if print_ars == 'y':
            print("Original array:", data)
            print("Centered data:", centered_data)
            
        return centered_data
    
    def multiple_regression(self, X, y):
        # Add a column of ones to the end of the X matrix
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        # Calculate the coefficients
        beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

        return beta

    # Function to generate a bell-shaped vector
    def generate_bell_vector(self, n, width, location, kurtosis=0, plot = 'y'):
        x = np.linspace(0, 1, n)
        y = np.exp(-0.5 * ((x - location) / width) ** 2)
        
        if kurtosis != 0:
            y = y ** kurtosis
        
        y /= np.sum(y)  # Normalize the vector to sum up to 1
        
        if plot == 'y':
            plt.scatter(x, y)
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Bell-Shaped Vector')
            plt.show()
            
        return y
        
    def numpy2coords(self, boolean_array, keep_vals:bool = False):
        # Get the coordinates of the True values in the boolean array
        coordinates = np.array(np.where(boolean_array))
        
        if keep_vals:
            # Get the values at the coordinates
            values = boolean_array[coordinates[0], coordinates[1], coordinates[2]]
            
            # Add the values as a fourth row to the coordinates
            coordinates = np.vstack([coordinates, values])
        
        # Transpose the coordinates to get them in the correct shape
        coordinates = coordinates.T
        
        return coordinates

    def coords2numpy(self, coordinates, shape, keep_vals:bool = False):
        # Create an array with the same shape as the original array
        array = np.zeros(shape, dtype=float if keep_vals else bool)
        
        if keep_vals:
            # Set the cells at the coordinates to their corresponding values
            array[tuple(coordinates[:,:3].astype('int').T)] = coordinates[:,3]
        else:
            # Set the cells at the coordinates to True
            # if coordinates
            # array[tuple(coordinates.T)] = True
                array[tuple(coordinates[:,:3].astype('int').T)] = True
        
        return array

    def find_common_rows(self, values_array, mask_array, keep_vals:bool = False):
        cols_vals = values_array.shape[1] - 1
        cols_mask = mask_array.shape[1] - 1
        set1 = {tuple(row[:cols_vals]): row[cols_vals] for row in values_array}
        set2 = set(map(tuple, mask_array[:,:cols_mask]))
        
        common_rows = np.array([list(x) + ([set1[x]] if keep_vals else []) for x in set1.keys() & set2])
        return common_rows    

    def sort_by_column(self, array, column_index, top_n):
        # Get the column
        column = array[:, column_index]

        # Get the indices that would sort the column
        sorted_indices = np.argsort(column)

        # Reverse the indices to sort in descending order and get the top_n indices
        top_indices = sorted_indices[::-1][:top_n]

        # Sort the entire array by these indices
        sorted_array = array[top_indices]

        return sorted_array

    # Function to return the voxel coordinates based on the parameter represented in the 4th column
    def filter_array_by_size(self, array, size_min, size_max):
        filtered_array = array[(array[:, 3] >= size_min) & (array[:, 3] <= size_max)]
        return filtered_array

    def ecc_angle_to_coords(self, ecc, angle, dim = 425):
        
        y = ((1 + dim) / 2) - (ecc * np.sin(np.radians(angle)) * (dim / 8.4)) #y in pix (c_index)
        x = ((1 + dim) / 2) + (ecc * np.cos(np.radians(angle)) * (dim / 8.4)) #x in pix (r_index)
        
        x = ecc * np.cos(np.radians(angle))
        y = ecc * np.sin(np.radians(angle))
        return x, y


    def voxname_for_xyz(self, xyz_to_voxname:np.array, x:int, y:int, z:int):
        # Create a boolean mask that is True for rows where the first three columns match val1, val2, val3
        mask = (xyz_to_voxname[:, 0] == x) & (xyz_to_voxname[:, 1] == y) & (xyz_to_voxname[:, 2] == z)

        # Use the mask to select the matching row(s) and the fourth column
        voxname = xyz_to_voxname[mask, 3]

        # If there is only one matching row, voxname will be a one-element array
        # You can get the element itself with:
        if voxname.size == 1:
            voxname = voxname[0]

        return voxname



# Utility function to visualize dictionary structures
def print_dict_structure(d, indent=0):
    for key, value in d.items():
        print(' ' * indent + str(key))
        if isinstance(value, dict):
            print_dict_structure(value, indent + 4)
            
            
def print_large(item):
    with np.printoptions(threshold=np.inf):
        print(item)
        
        
def get_zscore(data, print_ars = 'y'):
    mean_value = np.mean(data)
    std_dev = np.std(data)

    # Calculate z-scores
    z_scores = (data - mean_value) / std_dev

    if print_ars == 'y':
        print("Original array:", data)
        print("Z-scores:", z_scores)
        
    return z_scores

def cap_values(array = None, lower_threshold = None, upper_threshold = None):
    
    if upper_threshold is None:
        upper_threshold = np.max(array)
    else:
        # Identify values above the upper threshold
        above_upper_threshold = array > upper_threshold
        
        # Identify the highest value below the upper threshold
        highest_below_upper_threshold = array[array <= upper_threshold].max()

        # Replace values above the upper threshold with the highest value below the upper threshold
        array[above_upper_threshold] = highest_below_upper_threshold

    if lower_threshold is None:
        lower_threshold = np.min(array)
    else:
        # Identify values below the lower threshold
        below_lower_threshold = array < lower_threshold

        # Identify the lowest value above the lower threshold
        lowest_above_lower_threshold = array[array >= lower_threshold].min()

        # Replace values below the lower threshold with the lowest value above the lower threshold
        array[below_lower_threshold] = lowest_above_lower_threshold

    return array

def mean_center(data, print_ars = 'y'):
    mean_value = np.mean(data)

    # Mean centering
    centered_data = data - mean_value

    if print_ars == 'y':
        print("Original array:", data)
        print("Centered data:", centered_data)
        
    return centered_data
 
def multiple_regression(X, y):
    # Add a column of ones to the end of the X matrix
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    # Calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta

# Function to generate a bell-shaped vector
def generate_bell_vector(n, width, location, kurtosis=0, plot = 'y'):
    x = np.linspace(0, 1, n)
    y = np.exp(-0.5 * ((x - location) / width) ** 2)
    
    if kurtosis != 0:
        y = y ** kurtosis
    
    y /= np.sum(y)  # Normalize the vector to sum up to 1
    
    if plot == 'y':
        plt.scatter(x, y)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Bell-Shaped Vector')
        plt.show()
        
    return y
    
def numpy2coords(boolean_array, keep_vals:bool = False):
    # Get the coordinates of the True values in the boolean array
    coordinates = np.array(np.where(boolean_array))
    
    if keep_vals:
        # Get the values at the coordinates
        values = boolean_array[coordinates[0], coordinates[1], coordinates[2]]
        
        # Add the values as a fourth row to the coordinates
        coordinates = np.vstack([coordinates, values])
    
    # Transpose the coordinates to get them in the correct shape
    coordinates = coordinates.T
    
    return coordinates

def coords2numpy(coordinates, shape, keep_vals:bool = False):
    # Create an array with the same shape as the original array
    array = np.zeros(shape, dtype=float if keep_vals else bool)
    
    if keep_vals:
        # Set the cells at the coordinates to their corresponding values
        array[tuple(coordinates[:,:3].astype('int').T)] = coordinates[:,3]
    else:
        # Set the cells at the coordinates to True
        # if coordinates
        # array[tuple(coordinates.T)] = True
            array[tuple(coordinates[:,:3].astype('int').T)] = True
    
    return array

def find_common_rows(values_array, mask_array, keep_vals:bool = False):
    cols_vals = values_array.shape[1] - 1
    cols_mask = mask_array.shape[1] - 1
    set1 = {tuple(row[:cols_vals]): row[cols_vals] for row in values_array}
    set2 = set(map(tuple, mask_array[:,:cols_mask]))
    
    common_rows = np.array([list(x) + ([set1[x]] if keep_vals else []) for x in set1.keys() & set2])
    return common_rows    

def _sort_by_column(array, column_index, top_n):
    # Get the column
    column = array[:, column_index]

    # Get the indices that would sort the column
    sorted_indices = np.argsort(column)

    # Reverse the indices to sort in descending order and get the top_n indices
    top_indices = sorted_indices[::-1][:top_n]

    # Sort the entire array by these indices
    sorted_array = array[top_indices]

    return sorted_array

# Function to return the voxel coordinates based on the parameter represented in the 4th column
def filter_array_by_size(array, size_min, size_max):
    filtered_array = array[(array[:, 3] >= size_min) & (array[:, 3] <= size_max)]
    return filtered_array

def ecc_angle_to_coords(ecc, angle, dim = 425):
    
    y = ((1 + dim) / 2) - (ecc * np.sin(np.radians(angle)) * (dim / 8.4)) #y in pix (c_index)
    x = ((1 + dim) / 2) + (ecc * np.cos(np.radians(angle)) * (dim / 8.4)) #x in pix (r_index)
    
    x = ecc * np.cos(np.radians(angle))
    y = ecc * np.sin(np.radians(angle))
    return x, y


def _get_voxname_for_xyz(xyz_to_voxname, x, y, z):
    # Create a boolean mask that is True for rows where the first three columns match val1, val2, val3
    mask = (xyz_to_voxname[:, 0] == x) & (xyz_to_voxname[:, 1] == y) & (xyz_to_voxname[:, 2] == z)

    # Use the mask to select the matching row(s) and the fourth column
    voxname = xyz_to_voxname[mask, 3]

    # If there is only one matching row, voxname will be a one-element array
    # You can get the element itself with:
    if voxname.size == 1:
        voxname = voxname[0]

    return voxname

# Function to plot the hypotheses for the feature and unpredictability sensitivity
def hypotheses_plot(n_layers:int=5, bell_width:float=.0115, bell_loc:float=3.0, bell_kurtosis:float=.01):

    hypothesis_1 = hypothesis_2 = np.zeros((4, n_layers))
    for i in range(4):
        loc = i/bell_loc
        # generate_bell_vector(5, 0.15, loc)
        hypothesis_1[i, :] = np.array(generate_bell_vector(n_layers, bell_width, loc, bell_kurtosis, plot = 'n'))
        
    hypothesis_3 = np.zeros((4, n_layers))
    for i in range(4):
        loc = i/bell_loc
        # generate_bell_vector(5, 0.15, loc)
        hypothesis_3[np.abs(i - 3), :] = np.array(generate_bell_vector(n_layers, bell_width, loc, bell_kurtosis, plot = 'n'))
        
    # Keeping the same values for Hypothesis 4
    hypothesis_4 = np.zeros((4, n_layers))
    for i in range(4):
        hypothesis_4[i, :] = hypothesis_1[3]

    hypotheses = [hypothesis_1, hypothesis_2, hypothesis_3, hypothesis_4]

    # Visual areas
    visual_areas = ['V1', 'V2', 'V3', 'V4']

    # Define a gradient colormap from dark blue to light red
    # cmap = LinearSegmentedColormap.from_list('NavyBlueVeryLightGreyDarkRed', ['#000080', '#CCCCCC', '#FFA500', '#FF0000'], N=n_layers)
    # cmap = LinearSegmentedColormap.from_list('NavyBlueVeryLightGreyDarkRed', ['#000039', '#000080', '#CCCCCC', '#FFA000', '#FF0025', '#800000'], N=13)
    # cmap = LinearSegmentedColormap.from_list('NavyBlueVeryLightGreyDarkRed', ['#000039', '#000090', '#6699CC', '#90DEFF','#CBEAE8', '#E9E9E9', '#F5DEB3', '#FFD700', '#FFA500', '#FF4500', '#800000'], N=13)
    cmap = LinearSegmentedColormap.from_list(
        "NavyBlueVeryLightGreyDarkRed",
        [
            "#000039",
            "#0000C0",
            "#426CFF",
            "#8DC2FF",
            "#BDF7FF",
            "#E3E3E3",
            "#FFC90A",
            "#FF8B00",
            "#FF4D00",
            "#E90000",
            "#800000",
        ],
        N=n_layers,
    )

    # Plotting
    # fig, axs = plt.subplots(1, 4, figsize=(np.array([14, 4.25])*1.1), sharey=True)
    fig, axs = plt.subplots(1, 4, figsize=(np.array([14.5, 4])*1.1), sharey=True)
        
    plt.subplots_adjust(wspace=-1)  # Adjust this value to your liking

    for i, hypothesis in enumerate(hypotheses):
        ax = axs[i]
        # Transpose the hypothesis so that each vector becomes a column
        hypothesis_t = np.transpose(hypothesis)
        bottom = np.zeros(len(hypothesis_t[0]))
        for j, data in enumerate(hypothesis_t):
            ax.bar(range(len(data)), data, bottom=bottom, edgecolor='none', linewidth=0.5, label=f'{j+1}', color=cmap(j))
            bottom += data
        ax.set_title(['Visual feature sensitivity\n\n\n', 
                    'Hypothesis 1: \nTraditionally hierarchical\nunpredictability sensitivity\n', 
                    'Hypothesis 2: \nReversed hierarchical\nunpredictability sensitivity\n', 
                    'Hypothesis 3: \nNon-hierarchical\nunpredictability sensitivity\n'][i], 
                    fontweight='normal', family = 'sans-serif', fontsize = 16)
        ax.set_ylim(0, 1)  # Adjust the y-axis limits based on your data range
        ax.set_yticks([0, 0.5, 1])  # Set y-ticks
        ax.set_yticklabels([0, 50, 100], fontweight="bold", fontsize=14)  # Change y-tick labels
        ax.spines['top'].set_visible(False)  # Remove top border
        ax.spines['right'].set_visible(False)  # Remove right border
        ax.set_xticks(range(len(visual_areas)))  # Set x-ticks
        ax.set_xticklabels(visual_areas, fontweight="bold", fontsize=14)  # Set x-tick labels
        ax.grid(False)  # Remove grid

    # Remove 'Category' label from the bottom
    fig.text(0.5, 0, '', ha='center', va='center', fontsize=14)
    # fig.text(0.0, 0.43, 'Layer Assignment (%)', va='center', rotation='vertical', fontweight='normal', fontsize = 15)
    # fig.text(0.105, -.01, 'Visual Areas', ha='left', fontweight='normal', fontsize = 15)

    # axs[0].legend(title='CNN Layer', loc = 'upper center', bbox_to_anchor=(0.5, 1.27),
    #         ncol=n_layers, fancybox=False, shadow=False, fontsize = 11.5, columnspacing = .55)

    plt.tight_layout()
    plt.show()
    