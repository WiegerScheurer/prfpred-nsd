import os
import pickle
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore as zs
from tqdm.notebook import tqdm

# os.chdir('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/')
# sys.path.append('/Users/wiegerscheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/rfpred_local') #Otherwise it cannot find classes and such
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/nsdcode')

# import lgnpy.CEandSC.lgn_statistics
# from lgnpy.CEandSC.lgn_statistics import LGN, lgn_statistics, loadmat

from unet_recon.inpainting import UNet


class DataFetch():
    
    def __init__(self, NSPobject):
        self.nsp = NSPobject
        pass
    
    # REDUNDANT
    # Function to get the pRF-based voxel selections
    # IMPROVE: make sure that it also works for all subjects later on. Take subject arg, clean up paths.
    def prf_selections(self):
        prf_selection_paths = [
            './data/custom_files/subj01/prf_mask_center_strict.pkl',
            './data/custom_files/subj01/prf_mask_central_strict_l.pkl',
            './data/custom_files/subj01/prf_mask_central_halfloose.pkl',
            './data/custom_files/subj01/prf_mask_central_loose.pkl',
            './data/custom_files/subj01/prf_mask_periphery_strict.pkl'
        ]
        return {os.path.basename(file): self.fetch_file(file) for file in prf_selection_paths}
    
    # General file fetching function 
    def fetch_file(self, file_path:str):
        """
        General function to acquire saved data from various file types
        file_type: str, the types of files to be fetched, either features or prf_selections
        """
        _, ext = os.path.splitext(file_path)
        
        # Check if file is of .h5 type
        if ext == '.h5':
            with h5py.File(file_path, 'r') as hf:
                data = hf.keys()
                return {key: np.array(hf[key]).flatten() for key in data}
        # Check if file is of .pkl type
        elif ext == '.pkl':
            with open(file_path, 'rb') as fp:
                return pickle.load(fp)
        # Check if file is of .csv type
        elif ext == '.csv':
            return pd.read_csv(file_path)
    
    # Function to load in nifti (.nii.gz) data and create some useful variables 
    def get_dat(self, path:str):
        full_dat = nib.load(path)
        dat_array = full_dat.get_fdata()
        
        # Calculate the range of values
        flat_arr = dat_array[~np.isnan(dat_array)]
        dat_dim = dat_array.shape

        return full_dat, dat_array, dat_dim, {'min': round(np.nanmin(flat_arr),7), 'max': np.nanmax(flat_arr), 'mean': round(np.nanmean(flat_arr),5)}
    
    # # Hard coded, make flexible
    # # Function to load in all the computed predictability estimates, created using the get_pred.py and pred_stack.sh scripts.
    # def load_pred_estims(
    #     self,
    #     subject: str = None,
    #     start: int = None,
    #     n_files: int = None,
    #     verbose: bool = False,
    #     cnn_type: str = "alexnet",
    #     peripheral:bool = False,
    #     dense:bool=False,
    #     peri_ecc:float|None = None,
    #     peri_angle:int|None = None,
    # ):
    #     """
    #     Load predicted estimations from .h5 files.

    #     Args:
    #         subject (str, optional): The subject. Defaults to None.
    #         start (int, optional): The starting index. Defaults to None.
    #         n_files (int, optional): The number of files to load. Defaults to None.
    #         verbose (bool, optional): Whether to print loading progress. Defaults to False.
    #         cnn_type (str, optional): The type of CNN. Defaults to "alexnet".
    #         peripheral (bool, optional): Whether to load peripheral estimations. Defaults to False.
    #         peri_ecc (float, optional): The eccentricity of peripheral estimations. Defaults to None.
    #         peri_angle (int, optional): The angle of peripheral estimations. Defaults to None.

    #     Returns:
    #         list: A list of dictionaries containing the loaded estimations.
    #     """
        
    #     dict_list = []

    #     peri_str = f"/peripheral/ecc{peri_ecc}_angle{peri_angle}" if peripheral else ""
    #     dense_str = "dense/" if dense else ""
    #     datapath = f"{self.nsp.own_datapath}/visfeats{peri_str}/pred/{dense_str}"

    #     # Get a list of files in the directory
    #     files = os.listdir(datapath)

    #     # Filter files that start with "pred_payloads" and end with ".h5"
    #     filtered_files = [
    #         file
    #         for file in files
    #         if file.startswith("pred_payloads") and file.endswith(f"{cnn_type}.h5")
    #     ]

    #     # Sort files based on the first number after 'pred_payloads'
    #     sorted_files = sorted(
    #         filtered_files,
    #         key=lambda x: int("".join(filter(str.isdigit, x.split("pred_payloads")[1]))),
    #     )

    #     # Load in the .h5 files
    #     for file_no, file in enumerate(sorted_files):
    #         if verbose:
    #             print(f"Now loading file {file_no + 1} of {len(sorted_files)}")
    #         # load in back dictionary
    #         with h5py.File(f"{datapath}/{file}", "r") as hf:
    #             data = hf.keys()

    #             dict = {key: np.array(hf[key]) for key in data}

    #         dict_list.append(dict)

    #     return dict_list
    
    # Hard coded, make flexible
    # Function to load in all the computed predictability estimates, created using the get_pred.py and pred_stack.sh scripts.
    def load_pred_estims(
            self,
            verbose: bool = False,
            cnn_type: str = "alexnet",
            peripheral:bool = False,
            dense:bool=False,
            peri_ecc:float|None = None,
            peri_angle:int|None = None,
            pred_folder:str = "pred",
        ):
            """
            Load predicted estimations from .h5 files.

            Args:
                subject (str, optional): The subject. Defaults to None.
                start (int, optional): The starting index. Defaults to None.
                n_files (int, optional): The number of files to load. Defaults to None.
                verbose (bool, optional): Whether to print loading progress. Defaults to False.
                cnn_type (str, optional): The type of CNN. Defaults to "alexnet".
                peripheral (bool, optional): Whether to load peripheral estimations. Defaults to False.
                peri_ecc (float, optional): The eccentricity of peripheral estimations. Defaults to None.
                peri_angle (int, optional): The angle of peripheral estimations. Defaults to None.

            Returns:
                list: A list of dictionaries containing the loaded estimations.
            """
            
            dict_list = []

            peri_str = f"/peripheral/ecc{peri_ecc}_angle{peri_angle}" if peripheral else ""
            dense_str = "dense/" if dense else ""
            # datapath = f"{self.nsp.own_datapath}/visfeats{peri_str}/pred/{dense_str}"
            datapath = f"{self.nsp.own_datapath}/visfeats{peri_str}/{pred_folder}/{dense_str}"

            # Get a list of files in the directory
            files = os.listdir(datapath)

            # Filter files that start with "pred_payloads" and end with ".h5"
            filtered_files = [
                file
                for file in files
                if file.startswith("pred_payloads") and file.endswith(f"{cnn_type}.h5")
            ]

            # Sort files based on the first number after 'pred_payloads'
            sorted_files = sorted(
                filtered_files,
                key=lambda x: int("".join(filter(str.isdigit, x.split("pred_payloads")[1]))),
            )

            # Load in the .h5 files
            for file_no, file in enumerate(sorted_files):
                if verbose:
                    print(f"Now loading file {file_no + 1} of {len(sorted_files)}")
                # load in back dictionary
                with h5py.File(f"{datapath}/{file}", "r") as hf:
                    data = hf.keys()

                    dict = {key: np.array(hf[key]) for key in data}

                dict_list.append(dict)

            return dict_list

    
    def get_betas(self, subject:str,
              roi_masks:Dict[str, np.ndarray], 
              start_session:int, 
              n_sessions:int) -> None:
        """Function to get the HRF signal beta values for one visual cortex roi and one subject at a time.
            Beware: first 3 columns contain the xyz coordinates of the voxel, the rest contains the betas.
            
            To conjoin all 40 sessions: use the _stack_betas method.

        Args:
            subject (str): The subject
            roi_masks (Dict[str, np.ndarray]): The dictionary containing boolean masks for the viscortex rois
            start_session (int): The first session
            n_sessions (int): The amount of sessions to get the betas for
        """        
        betapath = f'{self.nsp.nsd_datapath}/nsddata_betas/ppdata/{subject}/func1mm/betas_fithrf_GLMdenoise_RR/'
        
        for session in range(start_session, start_session + n_sessions): # If start = 1 and n = 10 it goes 1 2 3 4 5 6 7 8 9 10
            print(f'Working on session: {session}')
            session_str = f'{session:02d}'
            session_data = nib.load(f"{betapath}betas_session{session_str}.nii.gz").get_fdata(caching='unchanged')

            for roi in roi_masks[subject].keys():
                print(f'Working on roi: {roi}')
                roi_mask = roi_masks[subject][roi]
                filtbet = session_data[roi_mask.astype(bool)]

                # Get the indices of the True values in the mask
                if session == 1:  # only get indices for the first session
                    x, y, z = np.where(roi_mask)
                    x = x.reshape(-1, 1)
                    y = y.reshape(-1, 1)
                    z = z.reshape(-1, 1)
                    voxbetas = np.concatenate((x, y, z, filtbet), axis=1)
                else:
                    voxbetas = filtbet
                print(f'Current size of voxbetas: {voxbetas.shape}')        

                # Create the directory if it doesn't exist
                save_dir = f'{self.nsp.own_datapath}/{subject}/betas/{roi[:2]}'
                os.makedirs(save_dir, exist_ok=True)

                np.save(f'{save_dir}/beta_stack_session{session_str}.npy', voxbetas)
                print(f'Saved beta_stack_session{session_str}.npy')

            del session_data
            
    # What I Now need to figure out is whether it is doable to just save the aggregated version of this, or 
    # that it's quick enough to just stack them on the spot.
    def _stack_betas(self, subject:str, roi:str, verbose:bool, n_sessions:int, save_stack:bool=False) -> np.ndarray:
        """Hidden method to stack the betas for a given subject and roi

        Args:
            subject (str): The subject to acquire the betas for
            roi (str): The region of interest
            verbose (bool): Print out the progress
            n_sessions (int): The amount of sessions for which to acquire the betas
            save_stack (bool): Whether or not to save the beta stack

        Returns:
            np.ndarray: A numpy array with dimensions (n_voxels, n_betas) of which the first 3 columns
                represent the voxel coordinates and the rest the betas for each chronological trial
        """      
        with tqdm(total=n_sessions, disable=not verbose) as pbar:
            for session in range(1, 1+n_sessions):
                session_str = f'{session:02d}'
                betapath = f'{self.nsp.own_datapath}/{subject}/betas/{roi}/'

                if session == 1:
                    init_sesh = np.load(f'{betapath}beta_stack_session{session_str}.npy')
                    stack = np.hstack((init_sesh[:,:3], self.nsp.utils.get_zscore(init_sesh[:,3:], print_ars='n')))
                else:
                    stack = np.hstack((stack,  self.nsp.utils.get_zscore(np.load(f'{betapath}beta_stack_session{session_str}.npy'), print_ars='n')))

                if verbose:
                    pbar.set_description(f'NSD session: {session}')
                    pbar.set_postfix({f'{roi} betas': f'{stack.shape} {round(self.nsp.utils.inbytes(stack)/1000000000, 3)}gb'}, refresh=True)

                pbar.update()
            if save_stack:
                np.save(f'{self.nsp.own_datapath}/{subject}/betas/{roi}/all_betas.npy', stack)
        return stack

    def _stack_scce(self, save_stack:bool=False, cut_outliers:bool=True, sc_cutoff:Optional[float]=None, ce_cutoff:Optional[float]=None):
        """Stack the raw computed scce values into a single dataframe and deal with the outliers,
            NaN, and -inf values.

        Args:
        - save_stack (bool, optional): Whether to save the stacked version or not. Defaults to False.
        - cut_outliers (bool, optional): Whether to cut the outliers or not. Defaults to True.
        - sc_cutoff (float, optional): An optional manual STD cutoff value for spatial coherence
        - ce_cutoff (float, optional): An optional manual STD cutoff value for contrast energy

        Returns:
        - pandas.DataFrame: The stacked values, including z-scored versions
        """        
        # Directory containing the files
        directory = '/home/rfpred/data/custom_files/visfeats/scce/raw/'

        # Get a list of all files in the directory that start with 'scce_dict_center_'
        files = [file for file in os.listdir(directory) if file.startswith('scce_dict_center_')]        
        
        # Function to extract the number from the filename
        def _extract_number(filename):
            match = re.search(r'scce_dict_center_(\d+)', filename)
            return int(match.group(1)) if match else float('inf')

        def _remove_naninf(df):
            df_noninf = df.replace([np.inf, -np.inf], 0)  # Replace both inf and -inf with 0
            df_nonan = df_noninf.fillna(0)  # Replace NaN values with 0
            return df_nonan

        # Sort the files based on the number
        files.sort(key=_extract_number)

        # Initialize an empty list to hold the dataframes
        dfs = []

        # Loop over the files
        for file in files:
            # Only process .pkl files
            if file.endswith('.pkl'):
                # Fetch the file
                data = self.fetch_file(os.path.join(directory, file))
                # Append the data to the list
                dfs.append(data)

        # Concatenate all dataframes vertically
        result = pd.concat(dfs, axis=0)
        result_cln = _remove_naninf(result)

        if cut_outliers:
            if sc_cutoff is None:
                sc_cutoff = 1
            if ce_cutoff is None:
                ce_cutoff = 5
            result_cln['sc'] = self.nsp.utils.std_dev_cap(result_cln['sc'], sc_cutoff)
            result_cln['ce'] = self.nsp.utils.std_dev_cap(result_cln['ce'], ce_cutoff)
        
        # Add z-scored columns
        result_cln['ce_z'] = zs(result_cln['ce'])
        result_cln['sc_z'] = zs(result_cln['sc'])

        if save_stack:
            result_cln.to_pickle(f'{self.nsp.own_datapath}/visfeats/scce/scce_stack.pkl')

        return result_cln
    
    

    def store_predestims(
        self,
        cnn_type:str, 
        peripheral:bool=False, 
        dense:bool=False,
        peri_ecc:float|None=None, 
        peri_angle:int|None=None,
        pred_folder:str="pred"):
        """
        Stores predicted estimations in a CSV file.

        Args:
            cnn_type (str): The type of CNN.
            peripheral (bool, optional): Whether to use peripheral patches. Defaults to False.
            peri_ecc (float, optional): The eccentricity of peripheral patches. Defaults to None.
            peri_angle (int, optional): The angle of peripheral patches. Defaults to None.

        Returns:
            pd.DataFrame: The exploded and reset dataframe of predicted estimations.
        """
        
        peri_str = f"/peripheral/ecc{peri_ecc}_angle{peri_angle}" if peripheral else ""
        dense_str = "dense/" if dense else ""
        datapath = f"{self.nsp.own_datapath}/visfeats{peri_str}/{pred_folder}/{dense_str}"
        
        predstack = pd.DataFrame(self.load_pred_estims(cnn_type=cnn_type, peripheral=peripheral, 
                                                       dense=dense, peri_ecc=peri_ecc, peri_angle=peri_angle,
                                                       pred_folder=pred_folder))
        # predstack = pd.DataFrame(load_pred_estims(cnn_type=cnn_type, peripheral=peripheral, peri_ecc=peri_ecc, peri_angle=peri_angle))
        
        # Convert the list for every batch into separate rows
        predstack_exploded = predstack.apply(lambda x: x.explode())
        
        # Reset indices to turn it into a whole
        predstack_exploded.reset_index(drop=True, inplace=True)
        
        # Save the dataframe to csv
        predstack_exploded.to_csv(
            f"{datapath}all_predestims_{cnn_type}.csv", index=False
        )
        
        return predstack_exploded
        
        
    def stack_loc_contrasts(self, filepath: str, save: bool = False, savename: str = ""):
        """
        Stack and concatenate CSV files from a given directory.

        Args:
            filepath (str): The path to the directory containing the CSV files.
            save (bool, optional): Whether to save the concatenated dataframe as a CSV file. Defaults to False.
            savename (str, optional): The name to use when saving the CSV file. Defaults to an empty string.

        Returns:
            None
        """

        files = [file for file in os.listdir(filepath) if file.endswith('0.csv')]

        def sort_key(file_name):
            number_part = file_name.split('.')[-2]  # Get the second last item after splitting at each period
            return int(number_part) if number_part.isdigit() else float('inf')

        files.sort(key=sort_key)

        # Read the CSV files and store them in a list
        dfs = [pd.read_csv(f'{filepath}/{file}', index_col=0) for file in files]

        # Concatenate the dataframes
        df = pd.concat(dfs, ignore_index=True)

        if save:
            save_str = "_" if savename == "" else f"_{savename}"
            df.to_csv(f"{filepath}/all{save_str}.csv")
        
        
        
    def tidy_peripheral_contrasts(self, eccs:list|None=None, angles:list|None=None):
        """
        Semi hard code stacking function for peripheral contrasts given eccentricities and angles.
        
        Args:
            eccs (list|None): List of eccentricities. Defaults to [1.2, 2.0] if None.
            angles (list|None): List of angles. Defaults to [90, 210, 330] if None.
        """
        eccs = [1.2, 2.0] if eccs is None else eccs
        angles = [90, 210, 330] if angles is None else angles

        for ecc in eccs:
            for angle in angles:
                path = f"{self.nsp.own_datapath}/visfeats/peripheral/ecc{ecc}_angle{angle}/raw_rmsscce"
                self.stack_loc_contrasts(path, save=True, savename="rmsscce")


    