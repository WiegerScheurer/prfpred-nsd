os.chdir("/project/3018078.02/rfpred_dccn")

import os
import json
from colorama import Fore, Style

# Load codebase_home from config
def get_codebase_home():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'rfpred_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get('codebase_home', os.getcwd())

codebase_home = get_codebase_home()
os.chdir(codebase_home)

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
        nsd_datapath: str = f"{codebase_home}/data/natural-scenes-dataset",
        own_datapath: str = f"{codebase_home}/data/custom_files",
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
