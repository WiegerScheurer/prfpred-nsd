import fnmatch
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator
from nilearn import plotting
from scipy import stats
from scipy.stats import zscore as zs
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from classes.voxelsieve import VoxelSieve


class Analysis:

    def __init__(self, NSPobj):
        """Initialize Analysis with NSPobj."""
        self.nsp = NSPobj

    def load_y(
        self,
        subject: str,
        roi: str,
        voxelsieve=VoxelSieve,
        n_trials: Union[int, str] = 30000,
        include_xyz: bool = False,
    ) -> np.ndarray:
        """
        Loads the y values for a given subject and region of interest (ROI).
        Args:
            subject (str): The subject.
            roi (str): The region of interest.
            voxelsieve (VoxelSieve): VoxelSieve instance used to select voxels.
            n_trials (int or str, optional): Number of trials to load. If 'all', loads all trials (up to 30000). Default is 30000.
            include_xyz (bool, optional): Whether to include the x, y, z coordinates in the output. Default is False.
        Returns:
            np.ndarray: The loaded y-matrix consisting of the HRF signal betas from the NSD.
        Raises:
            ValueError: If n_trials is greater than 30000.
        """
        if isinstance(n_trials, int) and n_trials > 30000:
            raise ValueError("n_trials cannot be greater than 30000.")
        start_column = 0 if include_xyz else 3
        n_trials = 30000 if n_trials == "all" else n_trials
        return (
            np.load(f"{self.nsp.own_datapath}/{subject}/betas/{roi}/all_betas.npy")[
                voxelsieve.vox_pick, start_column:
            ]
        )[:, :n_trials]

    def run_ridge_regression(
        self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0, fit_icept: bool = False
    ) -> Ridge:
        """
        Run a ridge regression model on the data.
        Args:
            X (np.ndarray): Independent variables with shape (n_trials, n_features)
            y (np.ndarray): Dependent variable with shape (n_trials, n_outputs)
            alpha (float, optional): Regularisation parameter. Defaults to 1.0.
            fit_icept (bool, optional): Whether to fit intercept. Defaults to False.
        Returns:
            Ridge: The fitted Ridge model object.
        """
        model = Ridge(alpha=alpha, fit_intercept=fit_icept)
        model.fit(X, y)
        return model

    def _get_coefs(self, model: Ridge) -> np.ndarray:
        """Return model coefficients."""
        return model.coef_

    def _get_r(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Correlation coefficient between the columns of a matrix as goodness of fit metric.
        Args:
            y_true (np.ndarray): True target/response vector.
            y_pred (np.ndarray): Predicted target/response vector.
        Returns:
            np.ndarray: Correlation coefficient for every response.
        """
        zs = lambda v: (v - v.mean(0)) / v.std(0)
        return (zs(y_pred) * zs(y_true)).mean(0)

    # Duplicate of _get_r, so remove get_r_numpy

    def score_model(
        self, X: np.ndarray, y: np.ndarray, model: Ridge, cv: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate model performance using cross-validation.
        Args:
            X (np.ndarray): Independent variables.
            y (np.ndarray): Dependent variables.
            model (Ridge): Ridge model to score.
            cv (int, optional): Number of cross-validation folds. Defaults to 5.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted values and correlation scores for each fold.
        """
        kf = KFold(n_splits=cv)
        y_hat = []
        cor_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_hat_fold = model_clone.predict(X_test)
            r_fold = self._get_r(y_test, y_hat_fold)
            y_hat.append(y_hat_fold)
            cor_scores.append(r_fold)
        y_hat = np.concatenate(y_hat)
        return y_hat, cor_scores

    def plot_brain(
        self,
        prf_dict: dict,
        roi_masks: dict,
        subject: str,
        brain_numpy: np.ndarray,
        cmap,
        glass_brain: bool = False,
        save_img: bool = False,
        img_path: str = "brain_image.png",
        lay_assign_plot: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        """Function to plot a 3D np.ndarray with voxel-specific values on an anatomical brain template of that subject.

        Args:
        - prf_dict (dict): The pRF dictionary
        - roi_masks (dict): The dictionary with the 3D np.ndarray boolean brain masks
        - subject (str): The subject ID
        - brain_numpy (np.ndarray): The 3D np.ndarray with voxel-specific values
        - glass_brain (bool, optional): Optional argument to plot a glass brain instead of a static map. Defaults to False.
        - save_img (bool, optional): Optional argument to save the image to a file. Defaults to False.
        - img_path (str, optional): The path where the image will be saved. Defaults to 'brain_image.png'.
        """
        brain_nii = nib.Nifti1Image(
            brain_numpy, self.nsp.cortex.anat_templates(prf_dict)[subject].affine
        )
        if glass_brain:
            display = plotting.plot_glass_brain(
                brain_nii,
                display_mode="ortho",
                colorbar=True,
                cmap=cmap,
                symmetric_cbar=False,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            display = plotting.plot_stat_map(
                brain_nii,
                bg_img=self.nsp.cortex.anat_templates(prf_dict)[subject],
                display_mode="ortho",
                colorbar=True,
                cmap=cmap,
                symmetric_cbar=False,
                vmin=vmin,
                vmax=vmax,
            )

        if lay_assign_plot:
            # New code to format colorbar ticks
            def format_tick(x, pos):
                return f"{x:.0f}"

            formatter = FuncFormatter(format_tick)

            if display._cbar:
                display._cbar.update_ticks()
                display._cbar.ax.yaxis.set_major_formatter(formatter)
                display._cbar.ax.yaxis.set_major_locator(
                    FixedLocator(np.arange(0, 6))
                )  # set ticks manually

        plt.show()

        if save_img:
            display.savefig(img_path)  # save figure to file

    def stat_on_brain(
        self,
        prf_dict: dict,
        roi_masks: dict,
        subject: str,
        stat: np.ndarray,
        xyzs: np.ndarray,
        glass_brain: bool,
        cmap,
        save_img: bool = False,
        img_path: Optional[str] = "/home/rfpred/data/custom_files",
    ):
        """Function to create a brain plot based on a specific statistic and the corresponding voxel coordinates.

        Args:
        - prf_dict (dict): The pRF dictionary
        - roi_masks (dict): The dictionary with the 3D np.ndarray boolean brain masks
        - subject (str): The subject ID
        - stat (np.ndarray): The statistic to plot on the brain
        - xyzs (np.ndarray): The voxel coordinates
        - glass_brain (bool, optional): Optional argument to plot a glass brain instead of a static map. Defaults to False.
        """
        n_voxels = len(xyzs)
        statmap = np.zeros((n_voxels, 4))
        for vox in range(n_voxels):
            # statmap[vox, :3] = (xyzs[vox][0][0], xyzs[vox][0][1], xyzs[vox][0][2]) # this is for the old xyzs
            statmap[vox, :3] = xyzs[vox]
            statmap[vox, 3] = stat[vox]

        brainp = self.nsp.utils.coords2numpy(
            statmap, roi_masks[subject]["V1_mask"].shape, keep_vals=True
        )

        self.plot_brain(
            prf_dict, roi_masks, subject, brainp, cmap, glass_brain, save_img, img_path
        )

    def plot_learning_curve(self, X, y, model=None, alpha=1.0, cv=5):
        if model is None:
            # Create and fit the model
            model = self.run_ridge_regression(X, y, alpha)

        # Initialize the KFold object
        kf = KFold(n_splits=cv)

        # Initialize a list to store the scores for each fold
        scores = []

        # For each fold...
        for train_index, test_index in kf.split(X):
            # Split the data into training and testing sets
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fit the model on the training data
            model.fit(X_train, y_train)

            # Predict the values for the testing data
            y_hat = model.predict(X_test)

            # Calculate the correlation for each column
            scores_fold = [
                self._get_r(y_test[:, i], y_hat[:, i]) for i in range(y_test.shape[1])
            ]

            # Append the average correlation for this fold to the list
            scores.append(scores_fold)

        # Plot the scores
        for i, scores_fold in enumerate(scores, start=1):
            # Scatter plot of individual scores
            plt.scatter([i] * len(scores_fold), scores_fold, color="blue", alpha=0.5)

            # Line plot of mean score
            plt.plot(i, np.mean(scores_fold), color="red", marker="o")

        plt.xlabel("Fold")
        plt.ylabel("Correlation Score")
        plt.title("Learning Curve")

        # Set x-axis to only show integer values
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.show()

    def plot_residuals(self, X, y, model=None, alpha=1.0):
        """Plot the residuals of the model, which is the difference between the actual y and the predicted y (y_hat)

        Args:
        - X (_type_): _description_
        - y (_type_): _description_
        - model (_type_, optional): _description_. Defaults to None.
        - alpha (float, optional): _description_. Defaults to 1.0.
        """
        if model is None:
            # Create and fit the model
            model = self.run_ridge_regression(X, y, alpha)

        # Get the predicted values
        y_hat = model.predict(X)

        # Calculate the residuals
        residuals = y - y_hat

        # Create a scatter plot of the predicted values and residuals
        plt.scatter(y_hat, residuals, alpha=0.3)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.show()

    def analysis_chain_slim(
        self,
        subject: str,
        ydict: Dict[str, np.ndarray],
        voxeldict: Dict[str, VoxelSieve],
        X: np.ndarray,
        alpha: float,
        cv: int,
        rois: list,
        X_alt: np.ndarray,
        fit_icept: bool = False,
        save_outs: bool = False,
        regname: str | None = "",  # remove
        plot_hist: bool = True,
        alt_model_type: str = "alternative model",  # remove
        save_folder: str | None = None,  # remove
        X_str: str | None = None,  # remove
    ) -> pd.DataFrame:
        """Function to run a chain of analyses on the input data for each of the four regions of interest (ROIs).
            Includes comparisons with an uninformative dependent variable X matrix (such as a shuffled
            version of the original X matrix), to assess the quality of the model in a relative way.
            Returns an array of which the first 3 columns contain the voxel coordinates (xyz) and the
            fourth contains the across cross validation fold mean correlation R scores between the actual
            and predicted dependent variables (y vs. y_hat).

        Args:
        - ydict (np.ndarray): The dictionary containing the dependent variables y-matrices for each ROI.
        - X (np.ndarray): The independent variables X-matrix.
        - alpha (float): The regularisation parameter of the Ridge regression model.
        - cv (int): The number of cross-validation folds.
        - rois (list): The list of regions of interest (ROIs) to analyse.
        - X_alt (np.ndarray): The alternative model's X-matrix to compare the model against.
        - fit_icept (bool, optional): Whether or not to fit an intercept. If both X and y matrices are z-scored
                it is highly recommended to set it to False, otherwise detecting effects becomes difficult. Defaults to False.
        - save_outs (bool, optional): Whether or not to save the outputs. Defaults to False.

        Returns:
        - np.ndarray: Object containing the voxel coordinates and the mean R scores for each ROI. This can be
                efficiently turned into a numpy array using NSP.utils.coords2numpy, which in turn can be converted
                into a nifti file using nib.Nifti1Image(np.array, affine), in which the affine can be extracted
                from a readily available nifti file from the specific subject (using your_nifti.affine).
        """
        r_values = {}
        r_uninformative = {}
        regcor_dict = {}  # Dictionary to store cor_scores
        regcor_dict["X"] = {}

        if save_outs:
            save_path = f"{self.nsp.own_datapath}/{subject}/results/{save_folder}"
            os.makedirs(save_path, exist_ok=True)

        # Calculate scores for the given X matrix
        for roi in rois:
            y = ydict[roi]
            model_og = self.run_ridge_regression(X, y, alpha=alpha, fit_icept=False)
            _, cor_scores = self.score_model(X, y, model_og, cv=cv)
            r_values[roi] = np.mean(cor_scores, axis=0)
            regcor_dict["X"][roi] = cor_scores  # Save cor_scores to dictionary

            xyz = voxeldict[roi].xyz
            # Get a vector with the roi name repeated for each voxel
            roi_vec = np.array([roi] * len(xyz)).reshape(-1, 1)

            this_coords = np.hstack(
                (xyz, roi_vec, np.array(r_values[roi]).reshape(-1, 1))
            )
            this_coefs = np.mean(model_og.coef_, axis=1).reshape(
                -1, 1
            )  # This is where I get the mean betas for each voxel
            this_unpred_coef = model_og.coef_[
                :, -1
            ]  # This is supposed to get the beta coefficient for the last regressor (the unpredictability features)

            if roi == "V1":
                coords = this_coords
                beta_coefs = this_coefs
                unpred_coefs = this_unpred_coef.reshape(-1, 1)
            else:
                coords = np.vstack((coords, this_coords))
                beta_coefs = np.vstack((beta_coefs, this_coefs))
                unpred_coefs = np.vstack(
                    (unpred_coefs, this_unpred_coef.reshape(-1, 1))
                )

        regcor_dict["X_shuffled"] = {}

        # Calculate scores for the uninformative/baseline X matrix
        for roi in rois:
            y = ydict[roi]
            model_comp = self.run_ridge_regression(
                X_alt, y, alpha=alpha, fit_icept=fit_icept
            )
            _, cor_scores = self.score_model(X_alt, y, model_comp, cv=cv)
            r_uninformative[roi] = np.mean(cor_scores, axis=0)
            regcor_dict["X_shuffled"][roi] = cor_scores  # Save cor_scores to dictionary

            this_uninf_coefs = np.mean(model_comp.coef_, axis=1).reshape(-1, 1)

            if roi == "V1":
                uninf_scores = r_uninformative[roi].reshape(-1, 1)
                uninf_coefs = this_uninf_coefs
            else:
                uninf_scores = np.vstack(
                    (uninf_scores, r_uninformative[roi].reshape(-1, 1))
                )
                uninf_coefs = np.vstack((uninf_coefs, this_uninf_coefs))

        delta_r_df = pd.DataFrame()

        for i, roi in enumerate(rois[:4]):
            # Calculate and store the delta-R values
            if roi == "V1":
                all_vox_delta_r = (r_values[roi] - r_uninformative[roi]).reshape(-1, 1)
            else:
                all_vox_delta_r = np.vstack(
                    (
                        all_vox_delta_r,
                        (r_values[roi] - r_uninformative[roi]).reshape(-1, 1),
                    )
                )

            this_delta_r = round(
                np.mean(r_values[roi]) - np.mean(r_uninformative[roi]), 5
            )  # TODO: is this the same as first delta_r /vox and then mean?

            delta_r_df[roi] = [this_delta_r]

            if plot_hist:
                if roi == "V1":  # Create a figure with 4 subplots
                    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                    # Flatten the axs array for easy iteration
                    axs = axs.flatten()

                # Underlay with the histogram of r_uninformative[roi] values
                axs[i].hist(
                    r_uninformative[roi],
                    bins=25,
                    edgecolor=None,
                    alpha=1,
                    label=alt_model_type,
                    color="black",
                )  # Was "burlywood"
                # Plot the histogram of r_values[roi] values in the i-th subplot
                axs[i].hist(
                    r_values[roi],
                    bins=25,
                    edgecolor="black",
                    alpha=0.75,
                    label=X_str,
                    color="orangered",
                )  # Was "dodgerblue"
                axs[i].set_title(f"{roi} delta-R: {this_delta_r}")
                axs[i].legend() if roi == "V1" else None

                if roi == "V4":  # Add title and display the figure
                    plt.suptitle(f"{regname}", fontsize=16)
                    plt.tight_layout()
                    plt.show()

                plt.savefig(f"{save_path}/{regname}_plot.png") if save_outs else None

        coords = np.hstack(
            (
                coords,
                uninf_scores,
                all_vox_delta_r.reshape(-1, 1),
                beta_coefs,
                uninf_coefs,
                unpred_coefs,
            )
        )  # also added beta coefficients as last column. very rough but works

        coords_df = pd.DataFrame(
            coords,
            columns=[
                "x",
                "y",
                "z",
                "roi",
                "R",
                "R_alt_model",
                "delta_r",
                "betas",
                "betas_alt_model",
                "beta_unpred",
            ],
        )

        (
            coords_df.to_csv(f"{save_path}/{regname}_regdf.csv", index=False)
            if save_outs
            else None
        )

        return coords_df

    def analysis_chain(
        self,
        subject: str,
        ydict: Dict[str, np.ndarray],
        voxeldict: Dict[str, VoxelSieve],
        X: np.ndarray,
        alpha: float,
        cv: int,
        rois: list,
        X_uninformative: np.ndarray,
        fit_icept: bool = False,
        save_outs: bool = False,
        regname: Optional[str] = "",
        plot_hist: bool = True,
        shuf_or_baseline: str = "s",
        save_folder: str | None = None,
        X_str: str | None = None,
    ) -> (np.ndarray, pd.DataFrame):
        """Function to run a chain of analyses on the input data for each of the four regions of interest (ROIs).
            Includes comparisons with an uninformative dependent variable X matrix (such as a shuffled
            version of the original X matrix), to assess the quality of the model in a relative way.
            Returns an array of which the first 3 columns contain the voxel coordinates (xyz) and the
            fourth contains the across cross validation fold mean correlation R scores between the actual
            and predicted dependent variables (y vs. y_hat).

        Args:
        - ydict (np.ndarray): The dictionary containing the dependent variables y-matrices for each ROI.
        - X (np.ndarray): The independent variables X-matrix.
        - alpha (float): The regularisation parameter of the Ridge regression model.
        - cv (int): The number of cross-validation folds.
        - rois (list): The list of regions of interest (ROIs) to analyse.
        - X_uninformative (np.ndarray): The uninformative X-matrix to compare the model against.
        - fit_icept (bool, optional): Whether or not to fit an intercept. If both X and y matrices are z-scored
                It is highly recommended to set it to False, otherwise detecting effects becomes difficult. Defaults to False.
        - save_outs (bool, optional): Whether or not to save the outputs. Defaults to False.

        Returns:
        - np.ndarray: Object containing the voxel coordinates and the mean R scores for each ROI. This can be
                efficiently turned into a numpy array using NSP.utils.coords2numpy, which in turn can be converted
                into a nifti file using nib.Nifti1Image(np.array, affine), in which the affine can be extracted
                from a readily available nifti file from the specific subject (using your_nifti.affine).
        """
        comp_X_str = "Shuffled model" if shuf_or_baseline == "s" else "Baseline model"
        X_str = "model" if X_str is None else X_str
        r_values = {}
        r_uninformative = {}
        regcor_dict = {}  # Dictionary to store cor_scores
        regcor_dict["X"] = {}
        # Calculate scores for the given X matrix
        for roi in rois:
            y = ydict[roi]
            model_og = self.run_ridge_regression(X, y, alpha=alpha, fit_icept=False)
            _, cor_scores = self.score_model(X, y, model_og, cv=cv)
            r_values[roi] = np.mean(cor_scores, axis=0)
            regcor_dict["X"][roi] = cor_scores  # Save cor_scores to dictionary

            xyz = voxeldict[roi].xyz
            this_coords = np.hstack((xyz, np.array(r_values[roi]).reshape(-1, 1)))
            this_coefs = np.mean(model_og.coef_, axis=1).reshape(-1, 1)
            if roi == "V1":
                coords = this_coords
                beta_coefs = this_coefs
            else:
                coords = np.vstack((coords, this_coords))
                beta_coefs = np.vstack((beta_coefs, this_coefs))

        regcor_dict["X_shuffled"] = {}
        # Calculate scores for the uninformative/baseline X matrix
        for roi in rois:
            y = ydict[roi]
            model_comp = self.run_ridge_regression(
                X_uninformative, y, alpha=alpha, fit_icept=fit_icept
            )
            _, cor_scores = self.score_model(X_uninformative, y, model_comp, cv=cv)
            r_uninformative[roi] = np.mean(cor_scores, axis=0)
            regcor_dict["X_shuffled"][roi] = cor_scores  # Save cor_scores to dictionary
            if roi == "V1":
                uninf_scores = r_uninformative[roi].reshape(-1, 1)
            else:
                uninf_scores = np.vstack(
                    (uninf_scores, r_uninformative[roi].reshape(-1, 1))
                )

        coords = np.hstack(
            (coords, uninf_scores, beta_coefs)
        )  # also added beta coefficients as last column. very rough but works
        delta_r_df = pd.DataFrame()

        for i, roi in enumerate(rois[:4]):
            # Calculate and store the delta-R values
            this_delta_r = round(
                np.mean(r_values[roi]) - np.mean(r_uninformative[roi]), 5
            )
            delta_r_df[roi] = [this_delta_r]

            if plot_hist:
                if roi == "V1":  # Create a figure with 4 subplots
                    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                    # Flatten the axs array for easy iteration
                    axs = axs.flatten()

                # Underlay with the histogram of r_uninformative[roi] values
                axs[i].hist(
                    r_uninformative[roi],
                    bins=25,
                    edgecolor=None,
                    alpha=1,
                    label=comp_X_str,
                    color="black",
                )  # Was "burlywood"
                # Plot the histogram of r_values[roi] values in the i-th subplot
                axs[i].hist(
                    r_values[roi],
                    bins=25,
                    edgecolor="black",
                    alpha=0.75,
                    label=X_str,
                    color="orangered",
                )  # Was "dodgerblue"
                axs[i].set_title(f"{roi} delta-R: {this_delta_r}")
                axs[i].legend() if roi == "V1" else None

                if roi == "V4":  # Add title and display the figure
                    plt.suptitle(f"{regname}", fontsize=16)
                    plt.tight_layout()
                    plt.show()

        if save_outs:
            if save_folder is None:
                if "unpred" in regname:
                    save_folder = "unpred"
                elif "baseline" in regname:
                    save_folder = "baseline"
                elif "encoding" in regname:
                    save_folder = "encoding"
                else:
                    save_folder = "various"

            save_path = f"{self.nsp.own_datapath}/{subject}/brainstats/{save_folder}"
            os.makedirs(save_path, exist_ok=True)

            # Save the delta_r_df to a file
            delta_r_df.to_pickle(f"{save_path}/{regname}_delta_r.pkl")

            plt.savefig(f"{save_path}/{regname}_plot.png")  # Save the plot to a file

            # Save cor_scores to a file
            np.save(
                f"{save_path}/{regname}_regcor_scores.npy", coords
            )  # Save the coords to a file

            print(
                f"Succesfully saved the outputs to {regname}_plot.png and {regname}_regcor_scores.npy"
            )

            # Save the regcor_dict to a file
            with open(f"{save_path}/{regname}_regcor_dict.pkl", "wb") as f:
                pickle.dump(regcor_dict, f)

        return coords, delta_r_df

    def load_regresults(
        self,
        subject: str,
        prf_dict: dict,
        roi_masks: dict,
        feattype: str,
        cnn_layer: Optional[int] = None,
        plot_on_viscortex: bool = True,
        plot_result: Optional[str] = "r",
        lowcap: float = 0,
        upcap: float = None,
        file_tag: str = "",
        verbose: bool = True,
        reg_folder: str = "",
    ):
        """Function to load in the results from the regressions.

        Args:
        - subject (str): The subject for which the regression results are to be loaded
        - prf_dict (dict): The pRF dictionary
        - roi_masks (dict): The dictionary with the 3D np.ndarray boolean brain masks
        - feattype (str): options: 'rms', 'ce', 'sc', 'unpred', 'alexunet', 'alexown'
        - cnn_layer (int): Optional, only necessary for the 'unpred, 'alexunet', and 'alexown' feature types
        - plot_on_viscortex (bool): Whether to plot the results on the V1 mask
        - plot_result (str): The result to plot, options: 'r', 'r_shuf', 'r_rel', 'betas'
        - lowcap (float): The lower cap for the colourmap when plotting the brain
        - upcap (float): The upper cap, idem.
        - file_tag (str): Optional tag to append to the file name
        - verbose (bool): Whether to print the coords dataframe
        Out:
        - cor_scores_dict (Dict): The dictionary that contains for both the actual X matrix and the shuffled X matrix the
            r correlation scores for every separate cv fold.
        - coords (pd.DataFrame): This dataframe contains the mean r and beta scores over all of the cross-validation folds
        """

        reg_str = f"{feattype}"
        if feattype in ["unpred", "alexunet", "alexown"] or cnn_layer is not None:
            if cnn_layer is None:
                raise ValueError(
                    "Please provide a cnn_layer number for the feature type you have chosen"
                )
            reg_str = f"{feattype}_lay{cnn_layer}{file_tag}"

        # This is the dictionary that contains for both the actual X matrix and the shuffled X matrix the
        # r correlation scores for every separate cv fold.
        with open(
            f"{self.nsp.own_datapath}/{subject}/brainstats/{reg_folder}/{reg_str}_regcor_dict.pkl",
            "rb",
        ) as f:
            # Structure: cor_scores_dict['X' or 'X_uninformative'][roi][cross-validation fold]
            cor_scores_dict = pickle.load(f)

        # This dataframe contains the mean scores over all of the cross-validation folds
        coords = pd.DataFrame(
            np.load(
                f"{self.nsp.own_datapath}/{subject}/brainstats/{reg_folder}/{reg_str}_regcor_scores.npy"
            ),
            columns=["x", "y", "z", "r", "r_shuf", "beta"],
        )

        if plot_result == "r":
            plot_val = 3
        elif plot_result == "r_shuf":
            plot_val = 4
        elif plot_result == "r_rel":
            plot_val = 6
            coords["r_rel"] = coords["r"] - coords["r_shuf"]
        elif plot_result == "betas":
            plot_val = 5

        if verbose:
            print(coords)

        if plot_on_viscortex:
            brain_np = self.nsp.utils.coords2numpy(
                np.hstack(
                    (
                        np.array(coords)[:, :3],
                        np.array(coords)[:, plot_val].reshape(-1, 1),
                    )
                ),
                roi_masks[subject]["V1_mask"].shape,
                keep_vals=True,
            )

            self.plot_brain(
                prf_dict,
                roi_masks,
                subject,
                self.nsp.utils.cap_values(np.copy(brain_np), lowcap, upcap),
                False,
                save_img=False,
                img_path="/home/rfpred/imgs/rel_scores_np.png",
            )

            self.nsp.cortex.viscortex_plot(
                prf_dict=prf_dict,
                vismask_dict=roi_masks,
                plot_param=None,
                subject=subject,
                upcap=upcap,
                lowcap=lowcap,
                inv_colour=False,
                cmap="RdGy_r",
                regresult=brain_np,
            )

        return cor_scores_dict, coords

    def plot_delta_r(
        self,
        subject: str,
        rois: list,
        cnn_type: str = "alex",
        file_tag: str = "",
        save_imgs: bool = False,
        basis_param: str = "betas",
        which_reg: str = "unpred",
    ):
        """Function to plot the delta r values that have resulted from regressions across different cnn layers.
        Works for the predictability estimates and for the encoding models.

        Args:
        - subject (str): The subject.
        - rois (list): List of ROIs to consider.
        - cnn_type (str, optional): The type of CNN model to use. Defaults to 'alex'.
        - file_tag (str, optional): Optional tag to append to the file name. Defaults to ''.
        - save_imgs (bool, optional): Whether to save the images. Defaults to False.
        - basis_param (str, optional): The basis for assigning layers. Defaults to 'betas', alternative is 'r' or 'delta_r'.
        - which_reg (str, optional): The type of regression to use. Defaults to 'unpred'.
        """
        first_lay = 0
        last_lay = 5 if cnn_type == "alex" or cnn_type == "alexnet" else 6

        if which_reg == "unpred":
            feattype = f"{cnn_type}_unpred"
        elif which_reg == "encoding":
            feattype = "allvox_alexunet"
            first_lay = 1
            last_lay = 5

        n_layers = last_lay - first_lay

        if basis_param == "betas":
            param_col = 5
        elif basis_param == "r":
            param_col = 3
        elif basis_param == "delta_r":
            param_col = 6

        for layer in range(first_lay, last_lay):  # Loop over the layers of the alexnet
            cnn_layer = (
                f"er{str(layer)}" if which_reg == "encoding" else f"{str(layer)}"
            )

            delta_r_layer = (
                pd.read_pickle(
                    f"{self.nsp.own_datapath}/{subject}/brainstats/{feattype}_lay{cnn_layer}{file_tag}_delta_r.pkl"
                )
                .values[0]
                .flatten()
            )
            if layer == first_lay:
                all_delta_r = delta_r_layer
            else:
                all_delta_r = np.vstack((all_delta_r, delta_r_layer))

        df = pd.DataFrame(all_delta_r, columns=rois)
        print(df)

        df.reset_index(inplace=True)

        # Melt the DataFrame to long-form or tidy format
        df_melted = df.melt("index", var_name="ROI", value_name="b")

        fig, ax = plt.subplots()

        # Create the line plot
        sns.lineplot(x="index", y="b", hue="ROI", data=df_melted, marker="o", ax=ax)

        ax.set_xticks(range(n_layers))  # Set x-axis ticks to be integers from 0 to 4
        ax.set_xlabel(f"{cnn_type} Layer")
        ax.set_ylabel("Delta R Value")
        ax.set_title(f"Delta R Value per {cnn_type} Layer")

        if save_imgs:
            # Save the plot
            fig.savefig(
                f"{self.nsp.own_datapath}/{subject}/brainstats/{feattype}_lay{cnn_layer}{file_tag}_delta_r_plot.png"
            )

        plt.show()

    def assign_layers_old(
        self,
        subject: str,
        prf_dict: dict,
        roi_masks: dict,
        rois: list,
        cmap,
        cnn_type: str = "alex",
        plot_on_brain: bool = True,
        file_tag: str = "",
        save_imgs: bool = False,
        basis_param: str = "betas",
        which_reg: str = "unpred",
        man_title: str | None = None,
        return_nifti: bool = False,
        first_lay: int | None = None,
        last_lay: int | None = None,
        direct_folder: str | None = None,
    ):
        """
        Assigns layers to voxels based on the maximum beta value across layers for each voxel.

        Args:
            subject (str): The subject.
            prf_dict (dict): Dictionary containing pRF model results.
            roi_masks (dict): Dictionary containing ROI masks.
            rois (list): List of ROIs to consider.
            cnn_type (str, optional): The type of CNN model to use. Defaults to 'alex'.
            plot_on_brain (bool, optional): Whether to plot the results on the brain. Defaults to True.
            file_tag (str, optional): Optional tag to append to the file name. Defaults to ''.
            save_imgs (bool, optional): Whether to save the images. Defaults to False.
            basis_param (str, optional): The basis for assigning layers. Defaults to 'betas', alternative is 'r' or 'delta_r'.
            which_reg (str, optional): The type of regression to use. Defaults to 'unpred'.
        """
        first_lay = 0
        last_lay = 5 if cnn_type == "alex" or cnn_type == "alexnet" else 6
        n_layers = last_lay - first_lay

        if basis_param == "betas":
            param_col = 5
        elif basis_param == "r":
            param_col = 3
        elif basis_param == "delta_r":
            param_col = 6

        if direct_folder is None:
            if which_reg == "unpred":
                feattype = f"{cnn_type}_unpred"
            elif which_reg == "encoding":
                feattype = "allvox_alexunet"
            elif which_reg == "encoding_smallpatch":
                feattype = "smallpatch_allvox_alexunet"

            for layer in range(
                first_lay, last_lay
            ):  # Loop over the layers of the alexnet
                cnn_layer = (
                    f"er{str(layer)}"
                    if which_reg == "encoding" or which_reg == "encoding_smallpatch"
                    else f"{str(layer)}"
                )
                cordict, coords = self.load_regresults(
                    subject,
                    prf_dict,
                    roi_masks,
                    feattype,
                    cnn_layer,
                    plot_on_viscortex=False,
                    plot_result="r",
                    file_tag=file_tag,
                    verbose=False,
                    reg_folder=which_reg,
                )

                coords["delta_r"] = (
                    coords["r"] - coords["r_shuf"]
                )  # Compute the delta_r values

                if layer == first_lay:
                    all_betas = np.hstack(
                        (
                            np.array(coords)[:, :3],
                            np.array(coords)[:, param_col].reshape(-1, 1),
                        )
                    )
                else:
                    all_betas = np.hstack(
                        (all_betas, np.array(coords)[:, param_col].reshape(-1, 1))
                    )
        else:
            layer = first_lay if first_lay is not None else 0

            # Get the list of filenames
            filenames = os.listdir(
                f"{self.nsp.own_datapath}/{subject}/brainstats/{direct_folder}"
            )

            # Sort the filenames based on the layer number
            sorted_filenames = sorted(filenames, key=self.nsp.utils._extract_layno)
            print(sorted_filenames)

            for layerfile in sorted_filenames:
                print(layerfile)
                if fnmatch.fnmatch(layerfile, "*regcor_dict.pkl"):
                    with open(
                        f"{self.nsp.own_datapath}/{subject}/brainstats/{direct_folder}/{layerfile}",
                        "rb",
                    ) as f:
                        cordict = pickle.load(f)
                if fnmatch.fnmatch(layerfile, "*regcor_scores.npy"):
                    coords_np = np.load(
                        f"{self.nsp.own_datapath}/{subject}/brainstats/{direct_folder}/{layerfile}"
                    )
                    coords = pd.DataFrame(
                        coords_np, columns=["x", "y", "z", "r", "r_shuf", "beta"]
                    )
                    del coords_np
                    print(f"Now looking at layer numero {layer}")

                    coords["delta_r"] = (
                        coords["r"] - coords["r_shuf"]
                    )  # Compute the delta_r values

                    if layer == first_lay:
                        all_betas = np.hstack(
                            (
                                np.array(coords)[:, :3],
                                np.array(coords)[:, param_col].reshape(-1, 1),
                            )
                        )
                    else:
                        all_betas = np.hstack(
                            (all_betas, np.array(coords)[:, param_col].reshape(-1, 1))
                        )
                    layer += 1

        for n_roi, roi in enumerate(rois):
            n_roivoxels = len(cordict["X"][roi][0])

            if roi == "V1":
                vox_of_roi = np.ones((n_roivoxels, 1))
            else:
                vox_of_roi = (
                    np.vstack((vox_of_roi, (np.ones((n_roivoxels, 1)) * (n_roi + 1))))
                ).astype(int)

        all_betas_voxroi = np.hstack((all_betas, vox_of_roi))[:, 3:]
        all_betas_voxroi[:, :n_layers] = stats.zscore(
            all_betas_voxroi[:, :n_layers], axis=0
        )
        print(all_betas_voxroi)

        # Get the index of the maximum value in each row, excluding the last column
        max_indices = (
            np.argmax(all_betas_voxroi[:, :-1], axis=1) + 1
        )  # Add 1 to the max_indices to get the layer number
        print(max_indices)
        barcmap = LinearSegmentedColormap.from_list(
            "NavyBlueVeryLightGreyDarkRed",
            ["#000080", "#CCCCCC", "#FFA500", "#FF0000"],
            N=n_layers,
        )

        # Create a new colourmap for the glass brain plot, as it has difficulty adapting the colourmap to non-symmetrical/positive values
        colors = np.concatenate(
            [barcmap(np.linspace(0, 1, 128)), barcmap(np.linspace(0, 1, 128))]
        )
        glass_cmap = ListedColormap(colors, name="double_lay_assign")

        # Create a DataFrame from the array
        df = pd.DataFrame(
            all_betas_voxroi,
            columns=[f"col_{i}" for i in range(all_betas_voxroi.shape[1])],
        )

        # Rename the last column to 'ROI'
        df.rename(columns={df.columns[-1]: "ROI"}, inplace=True)

        # Add the max_indices as a new column
        df["AlexNet layer"] = max_indices

        # Convert the 'ROI' column to int for plotting
        df["ROI"] = df["ROI"].astype(int)

        # Calculate the proportions of max_indices within each ROI
        df_prop = (
            df.groupby("ROI")["AlexNet layer"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )

        # Create a mapping from old labels to new labels
        roi_mapping = {1: "V1", 2: "V2", 3: "V3", 4: "V4"}

        # Change the labels on the x-axis
        df_prop.rename(index=roi_mapping, inplace=True)

        # Plot the proportions using a stacked bar plot
        ax = df_prop.plot(kind="bar", stacked=True, colormap=barcmap)

        # Add a y-axis label
        ax.set_ylabel("Layer assignment (%)")

        # Get current handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Reverse handles and labels
        handles, labels = handles[::-1], labels[::-1]

        # Create legend
        legend = plt.legend(
            handles,
            labels,
            title="CNN\nLayer",
            loc="center right",
            bbox_to_anchor=(1.15, 0.5),
            ncol=1,
            fancybox=False,
            shadow=False,
            fontsize=10,
        )

        if man_title is None:
            plt.title(f"Layer assignment {which_reg} {cnn_type} {basis_param}-based")
        else:
            plt.title(man_title)

        if save_imgs:
            save_path = f"{self.nsp.own_datapath}/{subject}/brainstats/{cnn_type}_unpred_layassign{file_tag}"
            # Save the plot
            plt.savefig(f"{save_path}.png")
        else:
            save_path = ""

        plt.show()
        if plot_on_brain:
            self.stat_on_brain(
                prf_dict,
                roi_masks,
                "subj01",
                max_indices,
                all_betas[:, :3].astype(int),
                glass_brain=True,
                cmap=glass_cmap,
                save_img=save_imgs,
                img_path=f"{save_path}_glassbrain.png",
            )

        if return_nifti:
            brain_coords = np.hstack(
                (all_betas[:, :3].astype(int), (max_indices.reshape(-1, 1)))
            )  # Earlier I had +1 here
            brain_np = self.nsp.utils.coords2numpy(
                brain_coords, roi_masks["subj01"]["V1_mask"].shape, keep_vals=True
            )
            brain_nii = nib.Nifti1Image(
                brain_np,
                affine=self.nsp.cortex.anat_templates(prf_dict)[subject].affine,
            )
            if save_imgs:
                nib.save(brain_nii, f"{save_path}.nii")
            return brain_nii

    def explained_var_plot(
        self,
        relu_layer: int,
        n_pcs: int,
        smallpatch: bool = False,
        modelname: str = "VGG",
    ):
        """Method to plot the explained variance ratio of the PCA instance created.

        Args:
        - relu_layer (int): Which ReLU layer of the AlexNet used for the encoding analyses
                to plot the explained variance ratio of. Options are 1, 2, 3, 4, and 5.
                These correspond to overall layers 1, 4, 7, 9, 11 in the CNN.
        - n_pcs (int): The number of principle components. Options are 1000 or 600.
        """
        smallpatch_str = "smallpatch_" if smallpatch else ""

        pca_instance = joblib.load(
            f"{self.nsp.own_datapath}/visfeats/cnn_featmaps/{modelname}/pca_{smallpatch_str}{relu_layer}_{n_pcs}pcs.joblib"
        )
        # Create a figure and a set of subplots
        fig, ax = plt.subplots()

        # Number of components
        n_components = np.arange(1, len(pca_instance.explained_variance_ratio_) + 1)
        cumulative_explained_variance_ratio = np.cumsum(
            pca_instance.explained_variance_ratio_
        )
        # Plot the explained variance ratio
        ax.bar(
            n_components,
            pca_instance.explained_variance_ratio_,
            alpha=0.5,
            align="center",
            label="individual explained variance",
        )

        # Plot the cumulative explained variance ratio
        ax.step(
            n_components,
            cumulative_explained_variance_ratio,
            where="mid",
            label="cumulative explained variance",
        )

        # Add a horizontal line at y=0.95
        ax.axhline(y=0.95, color="r", linestyle="--", label="95% explained variance")

        # Add labels and title
        ax.set_xlabel("Principal components")
        ax.set_ylabel("Explained variance ratio")
        ax.set_title(
            f"Explained variance ratio across all {n_pcs} PCs of layer {relu_layer}"
        )

        # Add a legend
        ax.legend(loc="best")

        # Show the plot
        plt.show()
