

import os
import json
import os
import json
import re
import sys
import math
from importlib import reload
from math import sqrt
from typing import Dict, Tuple, Union
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, List
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import zscore as zs

# Load codebase_home from config
def get_codebase_home():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'rfpred_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get('codebase_home', os.getcwd())

codebase_home = get_codebase_home()
os.chdir(codebase_home)
def get_codebase_home():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'rfpred_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get('codebase_home', os.getcwd())

codebase_home = get_codebase_home()
os.chdir(codebase_home)

# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/')
# sys.path.append('/Users/wiegerscheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/rfpred_local') #Otherwise it cannot find classes and such
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/nsdcode')

import classes.natspatpred

# from classes.utilities import _extract_layno
from classes.natspatpred import NatSpatPred, VoxelSieve
from unet_recon.inpainting import UNet

NSP = NatSpatPred()
NSP.initialise(verbose=False)


class RegData:
    def __init__(
        self,
        subject: str = "subj01",
        folder: Optional[str] = "unpred",
        model: str = "vgg-b",
        statistic: str = "delta_r",  # delta_r, R_alt_model, or R
        verbose: bool = False,
        skip_norm_lay: bool = False,
        single_file: bool = False,
        exclude_columns: Optional[List[str]] = None,
    ):
        self.subject = subject
        self.folder = folder
        self.model = model
        self.statistic = statistic
        self.verbose = verbose
        self.skip_norm_lay = skip_norm_lay
        self.single_file = single_file
        self.exclude_columns = exclude_columns or []
        self.cnn_layers = None
        self.df = None

        self._build_df()

    def _build_df(
        self,
        main_df: bool = True,
        add_xyz: bool = True,
    ):
        # Use self.subject, self.folder, etc. directly
        subject = self.subject
        folder = self.folder
        model = self.model
        statistic = self.statistic
        verbose = self.verbose
        skip_norm_lay = self.skip_norm_lay
        single_file = self.single_file
        exclude_columns = self.exclude_columns

# class RegData:
#     def __init__(
#         self,
#         subject: str = "subj01",
#         folder: str | None = "unpred",
#         model: str = "vgg-b",
#         statistic: str = "delta_r",  # delta_r, R_alt_model, or R
#         verbose: bool = False,
#         skip_norm_lay: bool = False,
#         single_file: bool = False,
#         exclude_columns: list = None,
#     ):
#         self.subject = subject
#         self.folder = folder
#         self.model = model
#         self.statistic = statistic
#         self.cnn_layers = None
#         self.verbose = verbose
#         self.skip_norm_lay = skip_norm_lay
#         self.single_file = single_file
#         self._build_df(subject, folder, model, statistic, verbose=verbose, skip_norm_lay=skip_norm_lay, single_file=single_file)

#     def _build_df(
#         self,
#         subject: str,
#         folder: str,
#         model: str,
#         statistic: str,
#         main_df: bool = True,
#         add_xyz: bool = True,
#         verbose: bool = True,
#         skip_norm_lay: bool = False,
#         single_file: bool = False,
#         exclude_columns: list = None,
#     ):
        # Directory containing the CSV files
        directory = f"{NSP.own_datapath}/{subject}/results/{folder}/"

        # List of filenames in the directory
        filenames = os.listdir(directory)

        # Empty DataFrame to store the results
        df = pd.DataFrame()

        # Variable to store the 'roi' column from the first file
        roi_column = None
        xyz_columns = None
        fileno = 0
        for filename in filenames:
            # Get the layer number from the filename
            layno = NSP.utils.get_layer_file(filename, "lay")
            # Check if the layer is relevant, skip if it is 0 when this is a normalisation layer we're not interested in
            relevant_layer = False if layno == 0 and skip_norm_lay == True else True
            if single_file:
                most_sparse_file = True if filename == model + f"_regdf.csv" else False
            else:
                most_sparse_file = True
            
            # Check if the filename starts with the model name
            # if filename.startswith(model) and filename.endswith(".csv") and relevant_layer:
            if filename.startswith(model) and filename.endswith(".csv") and relevant_layer and most_sparse_file:

                # layno = NSP.utils.get_layer_file(filename, "lay") if self.folder == "unpred" else fileno

                if verbose:
                    print(f"Processing file {filename} for layer {layno+1}")

                # Read the CSV file into a DataFrame
                file_df = pd.read_csv(os.path.join(directory, filename))

                # Store the 'roi' column from the first file
                if roi_column is None and "roi" in file_df.columns:
                    roi_column = file_df["roi"]

                if (
                    xyz_columns is None
                    and "x" in file_df.columns
                    and "y" in file_df.columns
                    and "z" in file_df.columns
                ):
                    xyz_columns = file_df[["x", "y", "z"]]

                if statistic == "delta_beta":
                    # Check if the necessary columns exist
                    if (
                        "betas_alt_model" in file_df.columns
                        and "betas" in file_df.columns
                    ):
                        # subtract the beta values from the baseline model
                        baseline = file_df["betas_alt_model"]
                        unpred_model = file_df["betas"]
                        file_df["delta_beta"] = unpred_model - baseline
                    else:
                        print(f"Missing necessary columns in file {filename}")
                        continue

                # Select the statistic column and rename it
                file_df = file_df[[statistic]].rename(
                    columns={statistic: f"{statistic}_{layno+1}"}
                )

                # Add the column to the result DataFrame
                if df.empty:
                    df = file_df
                else:
                    df = pd.concat([df, file_df], axis=1)

                fileno += 1

        # Sort the remaining columns
        df = df.sort_index(
            axis=1,
            key=lambda x: x.map(lambda y: int(re.findall(f"{statistic}_(\d+)", y)[0])),
        )

        # Add the 'roi' column to the DataFrame
        if roi_column is not None:
            df.insert(0, "roi", roi_column)
        # Add the xyz columns to the DataFrame
        if xyz_columns is not None and add_xyz:
            df = pd.concat([xyz_columns, df], axis=1)

        # Exclude specified columns if exclude_columns is provided
        if exclude_columns:
            # Ensure exclude_columns is a set for faster lookup
            exclude_set = set(exclude_columns)
            df = df[[col for col in df.columns if col not in exclude_set]]

        if main_df:
            self.df = df
            # Find the indices of the columns that start with the statistic name
            self.cnn_layers = [
                i for i, col in enumerate(self.df.columns) if col.startswith(statistic)
            ]
        else:

            return df

    def _stat_to_nifti(self, max_or_weighted: str = "weighted", verbose: bool = True):
        pass

    def _zscore(self, verbose: bool = True, copy_df: bool | pd.DataFrame = False):
        """
        Z-score the values in the DataFrame.
        """
        data = self.df if copy_df is False else copy_df

        data.iloc[:, self.cnn_layers] = zs(data.iloc[:, self.cnn_layers])
        if verbose:
            print(
                "\033[1mDataFrame changed:\033[0m Values z-scored. Check whether this is necessary for the current statistic."
            )

        return data if copy_df is not False else None

    def _normalize_per_voxel(
        self, verbose: bool = True, copy_df: bool | pd.DataFrame = False
    ):
        """
        Normalize the values by the maximum value for each voxel (row), ensuring that the minimum value is 0.
        """
        data = self.df if copy_df is False else copy_df

        df_min = data.iloc[:, self.cnn_layers].min(axis=1)
        # df_max = data.iloc[:, 1:].max(axis=1)
        df_max = data.iloc[:, self.cnn_layers].max(axis=1)

        data.iloc[:, self.cnn_layers] = (
            data.iloc[:, self.cnn_layers]
            .sub(df_min, axis=0)
            .div(df_max - df_min, axis=0)
        )
        if verbose:
            print(
                "\033[1mDataFrame changed:\033[0m Values normalised by dividing by the maximum value for each voxel (row), min values capped at 0."
            )

        return data if copy_df is not False else None

    def _scale_to_baseline(
        self, verbose: bool = True, copy_df: bool | pd.DataFrame = False
    ):
        """
        Scale the values to the baseline value for each ROI.
        NOTE: This method is not recommended for the delta-R statistic. As the delta-R value is already
        a product of two correlation values that are standardized already, it is not necessary to scale
        although it might seem intuitively necessary. Subtracting the baseline value from the delta-R value
        is a balance between simplicity and interpretability.
        """
        data = self.df if copy_df is False else copy_df

        baseline_data = self._build_df(
            self.subject,
            self.folder,
            self.model,
            statistic="R_alt_model",
            main_df=False,
        ).iloc[:, self.cnn_layers[0]]

        scaled_data = (
            data.iloc[:, self.cnn_layers]
            .div(baseline_data, axis=0)
            .mul(1 - baseline_data.pow(2), axis=0)
        )

        data.iloc[:, self.cnn_layers] = scaled_data

        if verbose:
            print(
                "\033[1mDataFrame changed:\033[0m Values scaled to the baseline value for each ROI."
            )

        return data if copy_df is not False else None

    def _weigh_mean_layer(
        self, verbose: bool = True, copy_df: bool | pd.DataFrame = False
    ):

        data = self.df if copy_df is False else copy_df

        # Normalize each row by the sum of its values
        df_normalized = data.iloc[:, self.cnn_layers].div(
            data.iloc[:, self.cnn_layers].sum(axis=1), axis=0
        )

        # Multiply each value by its column index
        weights = [(NSP.utils._extract_layno(col)) for col in df_normalized.columns]
        # if verbose:
        #     print(f"These are the weights for each column: {weights}")

        df_weighted = df_normalized.mul(weights, axis=1)

        # Calculate the mean of the weighted values and add it as a new column
        data["Mean Weighted Layer"] = round(df_weighted.mean(axis=1) * max(weights), 2)
        if verbose:
            print(
                "\033[1mDataFrame changed:\033[0m Added weighted means of the relative CNN-layer delta-R ranking to the DataFrame."
            )

        return data if copy_df is not False else None

    def _get_max_layer(
        self, verbose: bool = True, copy_df: bool | pd.DataFrame = False
    ):
        """
        Get the maximum layer index for each row in the DataFrame. Corresponds with the CNN-layer that has
        the highest delta-R value for each ROI, or any other statistic.

        Parameters:
        - verbose (bool): If True, print a message indicating that the maximum layer index has been added to the DataFrame.

        Returns:
        None
        """
        data = self.df if copy_df is False else copy_df

        # Get the index of the maximum value in each row
        max_indices = (
            np.argmax(data.values[:, self.cnn_layers], axis=1) + 1
        )  # Add 1 to the max_indices to get the layer number

        assign_str = "Max Layer"
        # Add the max_indices as a new column
        data[assign_str] = max_indices

        if verbose:
            print(
                "\033[1mDataFrame changed:\033[0m Added the maximum layer index to the DataFrame."
            )

        return data if copy_df is not False else None

    def _get_mean(self, verbose: bool = True, copy_df: bool | pd.DataFrame = False):
        """
        Get the mean value for each statistic row in the DataFrame.

        Parameters:
        - verbose (bool): If True, print a message indicating that the mean value has been added to the DataFrame.

        Returns:
        None
        """
        data = self.df if copy_df is False else copy_df

        # Get the index of the maximum value in each row
        mean_values = np.mean(data.values[:, self.cnn_layers], axis=1)

        assign_str = "Mean Statistic"

        # Add the max_indices as a new column
        data[assign_str] = mean_values

        if verbose:
            print(
                "\033[1mDataFrame changed:\033[0m Added the mean value to the DataFrame."
            )

        return data if copy_df is not False else None

    def assign_layers(
        self,
        max_or_weighted: str = "weighted",
        verbose: bool = True,
        title: str = None,
        input_df: pd.DataFrame = None,
        figsize: Tuple[float, float] = (6, 5.5),
        n_layers: int|None = None,
        legend:bool=True,
        smooth:bool=False,
        drop_V4:bool=False,
        save_at: str | None = None, # "auto"
    ):
        """
        Assigns layers to each ROI based on the maximum value in each row of a DataFrame.

        Parameters:
        - results (pd.DataFrame): The DataFrame containing the results.
        - max_or_weighted (str): The method to use for assigning layers. Default is 'max', other option is 'weighted'.
        - verbose (bool): If True, print a message indicating that the maximum layer index has been added to the DataFrame.
        - title (str): The title of the plot.
        - input_df (pd.DataFrame): The input DataFrame to use for plotting.
        - figsize (Tuple[float, float]): The size of the figure.
        - n_layers (int): The number of layers to assign.
        - legend (bool): If True, show the legend in the plot.
        - smooth (bool): If True, smooth the data.
        - drop_V4 (bool): If True, drop the V4 region from the DataFrame.
        - save_at (str | None): The path to save the plot. If None, do not save the plot. If "auto", save to the default location.
        
        Returns:
            dataframe: The DataFrame with the '{assign_str} Layer' column added.
        """

        df = self.df.copy() if input_df is None else input_df.copy()
        if verbose:
            print(
                "Using a copy of the DataFrame for layer assignment, the original DataFrame will not be changed."
            )
        df = self._normalize_per_voxel(verbose=verbose, copy_df=df)

        if max_or_weighted == "max":
            df = self._get_max_layer(verbose=verbose, copy_df=df)
            assign_str = "Max Layer"

        elif max_or_weighted == "weighted":
            df = self._weigh_mean_layer(verbose=verbose, copy_df=df)
            assign_str = "Mean Weighted Layer"

        lay_colours = max(self.cnn_layers) - self.cnn_layers[0] + 1 if n_layers is None else n_layers
        
        print(f"Number of layers: {lay_colours}") if verbose else None

        if drop_V4:
            df = df[df["roi"] != "V4"]

        barcmap = LinearSegmentedColormap.from_list(
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
            N=lay_colours,
        )
        
        if max_or_weighted == "max":
            # Calculate the proportions of max_indices within each ROI
            df_prop = (
                df.groupby("roi")[assign_str]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
            )
            # Ensure all categories are present
            df_prop = df_prop.reindex(columns=range(1, (lay_colours + 1)), fill_value=0)


        # THIS WEIGHTED IS A NICE IDEA, BUT THE LAYER INDICES BECOME SORT OF TRIVIAL
        # THE COLOURS DO NOT REALLY MATCH UP WITH THE MEAN VALUES, BECAUSE THE MEAN VALUES
        # OCCUPY A MUCH SMALLER RANGE THAN THE MAX VALUES, BUT THEY ARE STILL A BETTER INDICATOR
        # OF THE RELATIVE IMPORTANCE OF THE LAYERS, AND THE GRADIENT ALONG THE CORTICAL HIERARCHY
        elif max_or_weighted == "weighted":

            # Calculate the proportions of max_indices within each ROI
            df_prop = (
                df.groupby("roi")[assign_str]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
            )

        # Plot the proportions using a stacked bar plot
        ax = df_prop.plot(
            kind="bar",
            stacked=True,
            colormap=barcmap,
            edgecolor="none",
            width=0.8,
            figsize=figsize,
            legend=legend,
        )

        # Calculate the number of voxels in each ROI
        voxel_counts = df['roi'].value_counts()

        # Remove the existing x-axis labels
        ax.set_xticklabels([])

        # Add new x-axis labels with different font weights for the ROI label and the voxel count
        for i, label in enumerate(df_prop.index):
            ax.text(i, -0.075, label, fontsize=16, fontweight='bold', ha='center', transform=ax.get_xaxis_transform())
            ax.text(i, -0.125, f'({voxel_counts[label]})', fontsize=12, fontweight='normal', ha='center', transform=ax.get_xaxis_transform())

        # Add a y-axis label
        ax.set_ylabel("Layer assignment (%)", fontsize=20)
        ax.set_yticks([0, 0.5, 1])  # Set y-ticks
        ax.set_yticklabels(
            [0, 50, 100], fontsize=16, fontweight="bold"
        )  # Set y-tick labels
        ax.spines["top"].set_visible(False)  # Remove top border
        ax.spines["right"].set_visible(False)  # Remove right border
        plt.xticks(fontsize=16, fontweight="bold", rotation=0)

        leg_colours = [
            patches.Patch(
                color=barcmap(i / (len(self.cnn_layers))),
                label=str(layer - self.cnn_layers[0] + 1),
            )
            for i, layer in enumerate(self.cnn_layers)
        ]
        # if legend:
        #     # Create legend
        #     legend = plt.legend(
        #         handles=leg_colours,
        #         title="CNN\nLayer",
        #         loc="center right",
        #         bbox_to_anchor=(1.15, 0.5),
        #         ncol=1,
        #         fancybox=False,
        #         shadow=False,
        #         fontsize=10,
        #     )

        if legend:
            # Reverse the order of legend handles to flip the legend
            legend = plt.legend(
                handles=leg_colours[::-1],  # <-- reverse the list
                title="CNN\nLayer",
                loc="center right",
                bbox_to_anchor=(1.20, 0.5),
                ncol=1,
                fancybox=False,
                shadow=False,
                fontsize=10,
            )

        
        ax.set_xlabel("")
        
        if title is None:
            plt.title(
                f"Layer assignment {self.folder} {self.model} {self.statistic} {max_or_weighted}"
            )
        else:
            plt.title(title)

        if save_at is not None:
            if save_at == "auto":
                save_at = f"{NSP.own_datapath}/{self.subject}/results/{self.folder}/layassign/"
            else:
                save_at = save_at # os.path.join(save_at, f"{self.model}_{self.statistic}_mean_lines.png")
                # save_at = os.path.join(save_at, f"{self.model}_{self.statistic}_{max_or_weighted}_mean_lines.png")
            fig = plt.gcf()
            if not os.path.exists(os.path.dirname(save_at)):
                os.makedirs(os.path.dirname(save_at))
            # Save the figure
            fig.savefig(os.path.join(save_at, f"{self.model}_{self.statistic}_layer_assignment.png"), bbox_inches="tight", dpi=300)
        plt.show()

    def mean_lines(
        self,
        fit_polynom: bool = False,
        polynom_order: int = 2,
        verbose: bool = True,
        plot_catplot: bool = True,
        input_df: pd.DataFrame = None,
        title: str = None,
        fixed_ybottom: float | None = 0,
        fixed_ytop: float | None = None,
        log_y: bool = False,
        overlay: bool = False,
        fit_to: int = None,
        plot_ci:bool = True,
        # save_as: str | None = None,
        save_at: str | None = None,
    ):
        """
        Plots the mean values of each ROI across layers.

        Parameters:
        - fit_polynom (bool): Whether to fit a polynomial line to the data.
        - polynom_order (int): The order of the polynomial line to fit.
        - verbose (bool): Whether to print verbose output.
        - plot_catplot (bool): Whether to plot using a catplot.
        - input_df (pd.DataFrame): The input DataFrame to use for plotting.
        - title (str): The title of the plot.
        - fixed_ybottom (float | None): The fixed lower limit of the y-axis.
        - fixed_ytop (float | None): The fixed upper limit of the y-axis.
        - log_y (bool): Whether to use a logarithmic scale for the y-axis.
        - overlay (bool): Whether to overlay line plots on top of the catplot.
        - fit_to (int): The number of values to fit the line to.
        - plot_ci (bool): Whether to plot a confidence interval.
        - save_at (str | None): The path to save the plot. If None, do not save the plot. If "auto", save to the default location.

        Returns:
            None
        """
        
        df = self.df.copy() if input_df is None else input_df

        # Assuming df is your DataFrame
        present_id_vars = ("x", "y", "z", "roi", "Max Layer", "Mean Weighted Layer")

        # Create a tuple of column names that exist in both the DataFrame and the tuple
        present_id_vars = tuple(col for col in present_id_vars if col in df.columns)

        # Reshape the DataFrame
        df_melted = df.melt(
            # id_vars=present_id_vars, var_name="column", value_name="delta_r"
            id_vars=present_id_vars,
            var_name="column",
            value_name=self.statistic,
        )

        # Extract the integer from the column name
        def extract_number(col_name):
            match = re.search(r'delta_r_(\d+)', col_name)
            return int(match.group(1)) if match else float('inf')

        # Get the unique values in the 'column' field
        unique_values = df_melted["column"].unique()

        # Sort the unique values based on the extracted number
        unique_values_sorted = sorted(unique_values, key=extract_number)

        # Create a dictionary that maps each unique value to its rank order
        value_to_rank = {value: i for i, value in enumerate(unique_values_sorted)}
        
        # Replace the values in the 'column' field with their rank order
        df_melted["column"] = df_melted["column"].map(value_to_rank)

        df_melted = df_melted.sort_values(by="column")
        rois = sorted(df_melted["roi"].unique(), key=lambda x: int(x.split("V")[1]))

        # Create a color palette
        palette = sns.color_palette(n_colors=len(rois))

        # Create a dictionary mapping ROIs to colors
        roi_to_color = dict(zip(rois, palette))

        if plot_catplot:
            # Create a catplot with alpha set to 0.01
            catplot = sns.catplot(
                data=df_melted,
                x="column",
                y=self.statistic,
                hue="roi",
                jitter=True,
                palette=roi_to_color,
                alpha=0.01,
                legend=False,
            )
        else:
            # Create a figure and axes to plot on if not plotting catplot
            fig, ax = plt.subplots()

        # Create a regplot and a lineplot for each ROI
        for i, roi in enumerate(rois):
            roi_data = df_melted[df_melted["roi"] == roi]
            if fit_polynom:
                # Fit the line to the specified number of values
                roi_data = roi_data[roi_data["column"] < fit_to] if fit_to else roi_data
                sns.regplot(
                    data=roi_data,
                    x="column",
                    y=self.statistic,
                    scatter=False,
                    # scatter=overlay,
                    truncate=True,
                    order=polynom_order,
                    color=roi_to_color[roi],
                    ax=catplot.ax if plot_catplot else ax,
                    ci=plot_ci,  # Do not plot a confidence interval
                )
            elif overlay:  # Only plot the lineplot if overlay is True
                sns.lineplot(
                    data=roi_data,
                    x="column",
                    y=self.statifact,
                    color=roi_to_color[roi],
                    ax=catplot.ax if plot_catplot else ax,
                )
            else:
                sns.lineplot(
                    data=roi_data,
                    x="column",
                    y=self.statistic,
                    color=roi_to_color[roi],
                    ax=catplot.ax if plot_catplot else ax,
                )
            if save_at is not None:
                if save_at == "auto":
                    save_at = f"{NSP.own_datapath}/{self.subject}/results/{self.folder}/mean_lines/"
                else:
                    save_at = save_at # os.path.join(save_at, f"{self.model}_{self.statistic}_mean_lines.png")
                    # save_at = os.path.join(save_at, f"{self.model}_{self.statistic}_{max_or_weighted}_mean_lines.png")
                fig = plt.gcf()
                if not os.path.exists(os.path.dirname(save_at)):
                    os.makedirs(os.path.dirname(save_at))
                # Save the figure
                fig.savefig(os.path.join(save_at, f"{self.model}_{self.statistic}_mean_lines.png"), bbox_inches="tight", dpi=300)

        # Set the lower limit of the y-axis to 0
        ax = catplot.ax if plot_catplot else ax
        ax.set_ylim(bottom=fixed_ybottom, top=fixed_ytop)
        if log_y:
            ax.set_yscale('log')  # Set y-axis to logarithmic scale
        stat_label = "ΔR" if self.statistic == "delta_r" else self.statistic
        ax.set_ylabel(f"{stat_label} Value")
        ax.set_xlabel("CNN Layer")
        ax.set_xticks(range(len(self.cnn_layers)))
        ax.set_xticklabels([f"{i+1}" for i in range(len(self.cnn_layers))])

        # Manually add the legend
        (catplot.ax if plot_catplot else ax).legend(
            handles=[patches.Patch(color=roi_to_color[roi], label=roi) for roi in rois]
        )

        if title is not None:
            (catplot.ax if plot_catplot else ax).set_title(title)

        plt.show()
        
        # if save_as is not None:
        #     fig.savefig(save_as)

    def _delta_r_lines(self, cmap: str = "Reds"):
        """
        Method to plot the delta_r values for each voxel in each ROI across the layers of the CNN model.
        TODO: MAKE COMPATIBLE WITH THE FULL VGG MODEL OF 13 LAYERS, THE INDICES ARE MESSED UP (not true, it was correct all along)
        """

        model_str = "VGG-b" if self.model == "vgg-b" else "AlexNet"
        # regresults = RegData(folder="unpred", model=model, statistic="delta_r")
        regresults = self.df.copy()

        # regresults._weigh_mean_layer()
        df_reset = regresults.reset_index()

        rois = df_reset["roi"].unique()
        
        max_layer_index = 5 + len(self.cnn_layers)
        cols = df_reset.keys()[5:max_layer_index]
        # cols = df_reset.keys()[5:11]

        grid_size = math.ceil(math.sqrt(len(rois)))  # Calculate grid size
        fig, axs = plt.subplots(
            grid_size, grid_size, figsize=(10 * grid_size, 10 * grid_size)
        )
        fig.suptitle(
            f"Voxel specific ΔR across CNN layers for subject {self.subject[-1]}\n",
            fontsize=40,
            weight="normal",
        )
        # Get the colormap of desire
        # cmap = plt.get_cmap('Reds')
        cmap = plt.get_cmap(cmap)

        # Calculate global y min and max
        global_y_min = df_reset[cols].min().min()
        global_y_max = df_reset[cols].max().max()

        for i, roi in enumerate(rois):
            ax = axs[
                i // grid_size, i % grid_size
            ]  # Determine the position of the subplot
            data = df_reset[df_reset["roi"] == roi]
            voxels = data["index"].unique()
            for j, voxel in enumerate(voxels):
                voxel_data = data[data["index"] == voxel]
                # Use the colormap to get the color for this line
                color = cmap(j / len(voxels))
                ax.plot(
                    cols,
                    voxel_data[cols].values[0],
                    label=f"Voxel {voxel}",
                    color=color,
                )
            ax.set_title(f"{roi}", fontsize=30, weight="bold")  # Set title size
            ax.tick_params(
                axis="both", which="major", labelsize=25
            )  # Set tick label size
            ax.set_ylim([global_y_min, global_y_max])  # Set y min and max
            ax.set_xticklabels([col[-1] for col in cols], fontsize=25)
            ax.set_xlabel(f"{model_str} layer", fontsize=30)  # Set x label size
            ax.set_ylabel("ΔR", fontsize=30)  # Set y label size

        plt.tight_layout()
        plt.show()
