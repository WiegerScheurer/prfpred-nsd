#!/usr/bin/env python3

# This script efficiently pulls the betas from the NSD files using the updated get_betas method.

import os
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from matplotlib.ticker import MultipleLocator
import nibabel as nib
import pickle
from typing import Dict, Union, List
import argparse
import importlib
from importlib import reload

os.environ["OMP_NUM_THREADS"] = "10" # Limit the number of processors use to 10

os.chdir('/home/rfpred')
sys.path.append('/home/rfpred')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/')
sys.path.append('/home/rfpred/envs/rfenv/lib/python3.11/site-packages/nsdcode')

import classes.natspatpred
importlib.reload(classes.natspatpred)
from classes.natspatpred import NatSpatPred

NSP = NatSpatPred()
NSP.initialise()

predparser = argparse.ArgumentParser(description='Get the predictability estimates for a range of images of a subject')

predparser.add_argument('start', type=int, help='The starting index')
predparser.add_argument('n_sessions', type=int, help='The ending index')
predparser.add_argument('subject', type=str, help='The subject')

args = predparser.parse_args()
rois, roi_masks, viscortex_masks = NSP.cortex.visrois_dict(verbose=True)

# NSP.datafetch.get_betas(args.subject, viscortex_masks[args.subject], roi_masks, args.start, args.n_sessions)
NSP.datafetch.get_betas(args.subject, roi_masks, args.start, args.n_sessions)