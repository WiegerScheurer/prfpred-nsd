
import os


from colorama import Fore, Style


# from scipy.stats import zscore as zs
# from skimage import color
# from sklearn.base import clone
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import Lasso, LinearRegression, Ridge
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
# from sklearn.preprocessing import StandardScaler
# from torch.nn import Module
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from torchvision.models.feature_extraction import (create_feature_extractor,
#                                                    get_graph_node_names)
# from tqdm.notebook import tqdm

# os.chdir('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal')
# os.chdir('/Volumes/project/3018078.02/rfpred_dccn')
# os.chdir('/Users/wieger.scheurer/voorbeeld/')
os.chdir("/project/3018078.02/rfpred_dccn")

# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/')
# sys.path.append('/Users/wiegerscheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/Projects/rfpred_local') #Otherwise it cannot find classes and such
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/')
# sys.path.append('/Users/wiegerscheurer/miniconda3/envs/rfenv_minimal/lib/python3.11/site-packages/nsdcode')

### COMMETN OUT AS LONG AS THE MIGRATION IS INCOMPLETE, CHANGE BACK AT SOME POINT!!!
# import lgnpy.CEandSC.lgn_statistics
# from lgnpy.CEandSC.lgn_statistics import LGN, lgn_statistics, loadmat

from classes.analysis import Analysis
from classes.cortex import Cortex
from classes.datafetch import DataFetch
from classes.explorations import Explorations
from classes.stimuli import Stimuli
from classes.utilities import Utilities
from classes.voxelsieve import VoxelSieve
from unet_recon.inpainting import UNet


class NatSpatPred:

    def __init__(
        self,
        # nsd_datapath: str = "/home/rfpred/data/natural-scenes-dataset",
        # own_datapath: str = "/home/rfpred/data/custom_files",
        # nsd_datapath: str = "/Users/wiegerscheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/Projects/rfpred_local/data/natural-scenes-dataset",
        # own_datapath: str = "/Users/wiegerscheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/Projects/rfpred_local/data/custom_files",
        # nsd_datapath: str = "/Volumes/project/3018078.02/rfpred_dccn/data alias/natural-scenes-dataset",
        # own_datapath: str = "/Volumes/project/3018078.02/rfpred_dccn/data alias/custom_files",
        # nsd_datapath: str = "/Users/wieger.scheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/rfpred_local/data/natural-scenes-dataset",
        # own_datapath: str = "/Users/wieger.scheurer/Library/CloudStorage/OneDrive-RadboudUniversiteit/Donders/rfpred_local/data/custom_files",
        nsd_datapath: str = "/project/3018078.02/rfpred_dccn/data/natural-scenes-dataset",
        own_datapath: str = "/project/3018078.02/rfpred_dccn/data/custom_files",
    ):
        # Define the subclasses
        self.utils = None
        self.cortex = None
        self.stimuli = None
        self.datafetch = None
        self.explore = None
        self.analyse = None

        self.nsd_datapath = nsd_datapath
        self.own_datapath = own_datapath
        self.subjects = sorted(
            os.listdir(f"{nsd_datapath}/nsddata/ppdata/"),
            key=lambda s: int(s.split("subj")[-1]),
        )
        self.attributes = None
        self.hidden_methods = None

    # TODO: Expand this initialise in such way that it creates all the globally relevant attributes by calling on methods from the
    # nested classes
    def initialise(self, verbose: bool = True):
        self.utils = Utilities(self)
        self.cortex = Cortex(self)
        self.stimuli = Stimuli(self)
        self.datafetch = DataFetch(self)
        self.explore = Explorations(self)
        self.analyse = Analysis(self)

        self.attributes = [
            attr for attr in dir(self) if not attr.startswith("_")
        ]  # Filter out both the 'dunder' and hidden methods
        self.attributes_unfiltered = [
            attr for attr in dir(self) if not attr.startswith("__")
        ]  # Filter out only the 'dunder' methods
        if verbose:
            print(
                f"Naturalistic Spatial Prediction class: {Fore.LIGHTWHITE_EX}Initialised{Style.RESET_ALL}"
            )
            print("\nClass contains the following attributes:")
            for attr in self.attributes:
                print(f"{Fore.BLUE} .{attr}{Style.RESET_ALL}")
