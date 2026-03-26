# A hierarchy of spatial predictions across human visual cortex during natural vision

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19233920.svg)](https://doi.org/10.5281/zenodo.19233920)

### Repo Structure
The repository currently contains three main folders:

- *classes*: Contains class definitions used throughout the project.
    - Analysis: core functionality for analyses
    - Cortex: methods to work with neural data
    - Datafetch: tools for fetching any type of data
    - Explorations: functions for exploration
    - Regdata: regression-related functionality
    - Stimuli: tools to work with the natural scene stimuli
    - Utilities: general utilitiy functions
    - Voxelsieve: methods for voxel selection
    - Natspatpred: houses the parent class that has other classes as attribute
- *funcs*: Houses reusable functions for data processing and analysis.
- *scripts*: Includes scripts for running experiments, analyses, and generating results.

### Usage Notes
Not all scripts or functionality are required for the full empirical workflow. We are developing notebooks to highlight core functionalities, these will be released soon.

### Getting Started
Note that this codebase revolves around the Natural Scenes Dataset, due to which not all functionality can be replicated without it. It depends on the stimuli, fMRI BOLD — GLMsingle beta estimates, and most of the pRF parameter maps. 

> to be refined...