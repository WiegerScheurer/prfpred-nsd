from operator import truediv
import os
import sys

# os.chdir("/home/rfpred")
# sys.path.append("/home/rfpred")
# sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/")
# sys.path.append("/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode")

# os.chdir('/Users/wiegerscheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/Projects/rfpred_local/')

# os.chdir('/Users/wieger.scheurer/voorbeeld/')
# sys.path.append('/Users/wieger.scheurer/voorbeeld/nsdcode/nsdcode/')
# os.chdir('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/')
# sys.path.append('/Users/wiegerscheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/Projects/rfpred_local') #Otherwise it cannot find classes and such
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/nsdcode')
os.chdir("/project/3018078.02/rfpred_dccn")
sys.path.append("/project/3018078.02/rfpred_dccn/")
# import nsdcode
# from nsdcode.nsdcode import NSDmapdata, nsd_datalocation
# from nsdcode.nsdcode.nsd_datalocation import nsd_datalocation
# from nsdcode.nsdcode.nsd_mapdata import NSDmapdata
# from nsdcode.nsdcode.nsd_output import nsd_write_fs
# from nsdcode.nsdcode.utils import makeimagestack
import nibabel as nib
# import matplotlib.pyplot as plt
from nilearn import plotting
# import classes.natspatpred
from classes.natspatpred import NatSpatPred
# import classes.regdata
from classes.regdata import RegData

NSP = NatSpatPred()
NSP.initialise(verbose=False)


def reg_to_nifti(
    subject: str,
    reg_type: str,
    model: str,
    assign_stat: str = "max",
    reg_stat: str = "delta_r",
    plot_brain: bool = False,
    plot_lay_assign: bool = False,
    save_nifti: bool = True,
    mean_delta_r: bool = False,
    verbose: bool = False,
    peripheral: bool = False,
    peri_ecc:float|None=None,
    peri_angle:int|None=None,
    mean_unpred:bool=False,
    custom_subfolder:str|None=None,
    peri_suffix:str|None=None,
) -> None:
    
    os.makedirs(f"{NSP.own_datapath}/{subject}/stat_volumes", exist_ok=True)

    if "prf_dict" not in locals():
        rois, roi_masks, viscortex_masks = NSP.cortex.visrois_dict(verbose=False)
        prf_dict = NSP.cortex.prf_dict(rois, roi_masks)

    rd = RegData
    
    mean_unpred_str = "_mean_unpred" if mean_unpred else ""
    
    peri_str = f"/peri_ecc{peri_ecc}_angle{peri_angle}{mean_unpred_str}{peri_suffix}" if peripheral else ""
    peri_save_str = f"_peri_ecc{peri_ecc}_angle{peri_angle}{mean_unpred_str}{peri_suffix}" if peripheral else ""
    # This is to be able to retrieve files from folders that are not named exactly after their model (such as vggfull_gabor_baseline)
    folder_str = f"{model}{peri_str}" if custom_subfolder is None else custom_subfolder
    folder_save_str = custom_subfolder if custom_subfolder is not None else ""
    
    if reg_type == "unpred":
        results = rd(subject=subject, folder=f"{reg_type}/{folder_str}", model=model, statistic=reg_stat, skip_norm_lay=True)
    else:
        results = rd(subject=subject, folder=f"{reg_type}", model=model, statistic=reg_stat, skip_norm_lay=True)
        
    mean_stats = ["betas", "delta_beta", "beta_unpred", "R_alt_model", "R"]
        
    # if reg_stat == "betas" or reg_stat == "delta_beta" or reg_stat == "beta_unpred" or reg_stat == "R_alt_model":
    if reg_stat in mean_stats or mean_delta_r is True:
        results._get_mean(verbose=False)
        stat_str = "Mean Statistic"
        plot_lay_assign = False
        assign_stat = reg_stat
        
    else:
        if mean_delta_r:
            results._get_mean(verbose=False)
            plot_lay_assign = False
            stat_str = "Mean Statistic"
            assign_stat = "mean_delta_r"
        
        results._normalize_per_voxel(verbose=False)
        results._weigh_mean_layer(verbose=False)
        results._get_max_layer(verbose=False)
        
        if assign_stat == "max":
            stat_str = "Max Layer"
        elif assign_stat == "weighted":
            stat_str = "Mean Weighted Layer"

    new_df = results.df[["x", "y", "z", stat_str]]
    
    if verbose:
        print(results.df)

    lay_assign_str = "" if stat_str == "Mean Statistic" else "_layassign"

    NSP.utils.coords2nifti(
        subject,
        prf_dict,
        new_df.values,
        keep_vals=True,
        save_nifti=save_nifti,
        save_path=f"{NSP.own_datapath}/{subject}/stat_volumes/{reg_type}{peri_save_str}{folder_save_str}_{model}{lay_assign_str}_{assign_stat}.nii",
    )

    if plot_lay_assign:
        results.assign_layers(
            max_or_weighted=assign_stat,
            title=f"{stat_str} Layer Assignment of Voxels Across Visual Cortex\n{model}, ΔR based (Baseline vs. Baseline + Unpredictability)",
        )

    if plot_brain: # Only works when you save the nifti file
        # visualise the nifti
        img = nib.load(
            f"{NSP.own_datapath}/{subject}/stat_volumes/{reg_type}{peri_save_str}{folder_save_str}_{model}{lay_assign_str}_{assign_stat}.nii"
        )

        plotting.plot_stat_map(
            img,
            threshold=0.5,
            display_mode="ortho",
            cut_coords=(0, 0, 0),
            title=f"{stat_str} Layer Assignment of Voxels Across Visual Cortex\n{model}, ΔR based (Baseline vs. Baseline + Unpredictability)",
        )

def vol_to_surf(
    subject: str,
    source_file_name: str,
    sourcespace: str = "func1pt0",
    surface_type: str = "pial",
    interpmethod: str = "cubic",
    custom_path: str|None = None):
    """
    Convert a volume to a surface representation using NSDmapdata.

    Args:
        subject (str): The subject identifier.
        source_file_name (str): The name of the source file.
        sourcespace (str, optional): The source space. Defaults to "func1pt0".
        surface_type (str, optional): The surface type. Defaults to "pial".
        interpmethod (str, optional): The interpolation method. Defaults to "cubic".
    """
    base_path = NSP.nsd_datapath
    subjix = int(subject[-1])

    os.makedirs(f"{NSP.own_datapath}/{subject}/stat_surfaces", exist_ok=True)

    if source_file_name.endswith(".nii"):
        source_file_name = source_file_name[:-4]

    for hemisphere in ["lh", "rh"]:
        # initiate NSDmapdata
        nsd = NSDmapdata(base_path)

        nsd_dir = nsd_datalocation(base_path=base_path)
        nsd_betas = nsd_datalocation(base_path=base_path, dir0="betas")
        
        sourcedata = f"{NSP.own_datapath}/{subject}/stat_volumes/{source_file_name}.nii" if custom_path is None else custom_path
        sourcespace = "func1pt0"
        targetspace = f"{hemisphere}.{surface_type}"  # lh.pial and rh.pial are needed for unfolding the cortex

        targetdata = nsd.fit(
            subjix,
            sourcespace,
            targetspace,
            sourcedata,
            interptype=interpmethod,
            badval=0,
            outputfile=f"{NSP.own_datapath}/{subject}/stat_surfaces/{source_file_name}_{sourcespace}-{targetspace}-{interpmethod}.mgz",
            fsdir=f"{NSP.nsd_datapath}/nsddata/freesurfer/{subject}",
        )

        nsd.fit(
            subjix=subjix,
            sourcedata=sourcedata,
            sourcespace=sourcespace,
            targetspace=targetspace,
            interptype=interpmethod,
        )
