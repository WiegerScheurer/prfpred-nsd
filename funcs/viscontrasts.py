import warnings
import cv2
import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy
import yaml
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression, Ridge
from matplotlib.backends.backend_pdf import PdfPages
import mat73
from scipy import io

# This does not work like this, better remove
def lgn_statistics(im, file_name:str, threshold_lgn, coc:bool=True, config=None, 
                   verbose_filename:bool = True, verbose: bool = False, 
                   compute_extra_statistics: bool = False, crop_masks: list = [], 
                   force_recompute:bool=False, cache:bool=True, home_path:str='/home/niklas', 
                   ToRGC=lambda x: x, fov_imsize:tuple=None, result_manager=None):

    # result_manager = ResultManager(root=f'{home_path}/projects/lgnpy/cache', verbose=False)

    lgn = LGN(config=config, default_config_path=f'{home_path}/lgnpy/lgnpy/CEandSC/default_config.yml')

    if verbose_filename:
        print(f"Computing LGN statistics for {file_name}")
    # Check if file exists
    file_name = f"results_{file_name}.npz"
    if file_name is not None and not force_recompute and result_manager is not None:
        try:
            results = result_manager.load_result(filename=file_name)
        except:
            results = None
    else:
        results = None
    
    if type(im) is str:
        im = cv2.imread(im)

    #
    # Set image parameters
    #

    if im.shape[-1] == 2:
        IMTYPE = 1  # Gray
    elif im.shape[-1] == 3:
        IMTYPE = 2  # Color
    else:
        IMTYPE = 1
        # im = im.reshape((im.shape) + (1,))
        # print(im.shape) 

    if fov_imsize is None:
        fov_imsize = im.shape[:2]

    imsize = im.shape[:2]

    viewing_dist = lgn.get_attr('viewing_dist')
    imfovbeta, imfovgamma = get_field_of_view(lgn=lgn, imsize=fov_imsize, viewing_dist=viewing_dist)
    imfovbeta = ToRGC(imfovbeta).astype(int)
    imfovgamma = ToRGC(imfovgamma).astype(int)

    # We need adjusted imfovbeta, imfovgamma for the crops
    imfovbeta_crops = [[] for _ in range(len(crop_masks))]
    imfovgamma_crops = [[] for _ in range(len(crop_masks))]
    for index, mask in enumerate(crop_masks):
        _x = mask.sum(axis=0)
        mask_height = _x[_x > 0][0]
        _x = mask.sum(axis=1)
        mask_width = _x[_x > 0][0]
        mask_imsize = (mask_height, mask_width)

        mask_viewing_dist = np.mean(np.array(mask_imsize) / np.array(imsize)) * viewing_dist
        _imbeta, _imgamma = get_field_of_view(lgn=lgn, imsize=mask_imsize, viewing_dist=mask_viewing_dist)
        imfovbeta_crops[index] = _imbeta
        imfovgamma_crops[index] = _imgamma


    # (color_channels, (full+boxes), center-peripherie)
    ce = np.zeros((im.shape[-1], 1+len(crop_masks), 2))
    sc = np.zeros((im.shape[-1], 1+len(crop_masks), 2))
    beta = np.zeros((im.shape[-1], 1+len(crop_masks), 2))
    gamma = np.zeros((im.shape[-1], 1+len(crop_masks), 2))

    par1, par2, par3, mag1, mag2, mag3 = get_edge_maps(im=im, file_name=file_name, threshold_lgn=threshold_lgn, verbose=verbose, force_recompute=force_recompute, cache=cache, result_manager=result_manager, lgn=lgn, results=results, IMTYPE=IMTYPE, imsize=imsize)

    ##############
    # Compute Feature Energy and Spatial Coherence
    ##############

    def get_crop_masks(fov, mask):
        _start_x = mask.sum(axis=1).argmax()
        _start_y = mask.sum(axis=0).argmax()
        c_mask = np.zeros(mask.shape, dtype=np.bool8)
        c_mask[_start_x:_start_x+fov.shape[0], _start_y:_start_y+fov.shape[1]] = fov * mask[mask].reshape(fov.shape)
        c_mask_peri = np.zeros(mask.shape, dtype=np.bool8)
        c_mask_peri[_start_x:_start_x+fov.shape[0], _start_y:_start_y+fov.shape[1]] = (~fov) * mask[mask].reshape(fov.shape)
        return c_mask, c_mask_peri

    if verbose:
        print("Compute CE")

    magnitude = np.abs(par1[imfovbeta])
    # Full scene, red/gray
    ce[0, 0, 0] = np.mean(magnitude)
    if IMTYPE == 2:
        magnitude = np.abs(par2[imfovbeta])
        ce[1, 0, 0] = np.mean(magnitude)
        magnitude = np.abs(par3[imfovbeta])
        ce[2,0,0] = np.mean(magnitude)

    if compute_extra_statistics:
        # Peripherie
        peri = np.mean(np.abs(par1[~imfovbeta]))
        ce[0, 0, 1] = peri

        if IMTYPE == 2:
            peri = np.mean(np.abs(par2[~imfovbeta]))
            ce[1, 0, 1] = peri
            peri = np.mean(np.abs(par3[~imfovbeta]))
            ce[2, 0, 1] = peri
        
        # Custom boxes (crops)
        for mask_index, mask in enumerate(crop_masks):
            c_mask, c_mask_peri = get_crop_masks(imfovbeta_crops[mask_index], mask)

            box_center = np.mean(np.abs(par1[c_mask]))
            ce[0, mask_index+1, 0] = box_center
            box_peri = np.mean(np.abs(par1[c_mask_peri]))
            ce[0, mask_index+1, 1] = box_peri

            if IMTYPE == 2:
                box_center = np.mean(np.abs(par2[c_mask]))
                ce[1, mask_index+1, 0] = box_center
                box_peri = np.mean(np.abs(par2[c_mask_peri]))
                ce[1, mask_index+1, 1] = box_peri
                box_center = np.mean(np.abs(par3[c_mask]))
                ce[2, mask_index+1, 0] = box_center
                box_peri = np.mean(np.abs(par3[c_mask_peri]))
                ce[2, mask_index+1, 1] = box_peri


    if verbose:
        print("Compute SC")
    magnitude = np.abs(mag1[imfovgamma])
    sc[0,0,0] = np.mean(magnitude) / np.std(magnitude)
    if IMTYPE == 2:
        magnitude = np.abs(mag2[imfovgamma])
        sc[1,0,0] = np.mean(magnitude) / np.std(magnitude)
        magnitude = np.abs(mag3[imfovgamma])
        sc[2,0,0] = np.mean(magnitude) / np.std(magnitude)

    if compute_extra_statistics:
        # Peripherie
        peri = np.abs(mag1[~imfovgamma])
        sc[0, 0, 1] = np.mean(peri) / np.std(peri)

        if IMTYPE == 2:
            peri = np.abs(mag2[~imfovgamma])
            sc[1, 0, 1] = np.mean(peri) / np.std(peri)
            peri = np.abs(mag3[~imfovgamma])
            sc[2, 0, 1] = np.mean(peri) / np.std(peri)
        
        # Custom boxes (crops)
        for mask_index, mask in enumerate(crop_masks):
            c_mask, c_mask_peri = get_crop_masks(imfovgamma_crops[mask_index], mask)

            box_center = np.abs(mag1[c_mask])
            sc[0, mask_index+1, 0] = np.mean(box_center) / np.std(box_center)
            box_peri = np.abs(mag1[c_mask_peri])
            sc[0, mask_index+1, 1] = np.mean(box_peri) / np.std(box_peri)

            if IMTYPE == 2:
                box_center = np.abs(mag2[c_mask])
                sc[1, mask_index+1, 0] = np.mean(box_center) / np.std(box_center)
                box_peri = np.abs(mag2[c_mask_peri])
                sc[1, mask_index+1, 1] = np.mean(box_peri) / np.std(box_peri)
                box_center = np.abs(mag3[c_mask])
                sc[2, mask_index+1, 0] = np.mean(box_center) / np.std(box_center)
                box_peri = np.abs(mag3[c_mask_peri])
                sc[2, mask_index+1, 1] = np.mean(box_peri) / np.std(box_peri)

    #################
    # Compute Weibull parameters beta and gamma
    #################

    if verbose:
        print("Compute Weibull parameters beta")

    # n_bins = 1000
    n_bins = lgn.get_attr('n_bins_weibull')
    magnitude = np.abs(par1[imfovbeta])
    ax, h = lgn.create_hist(magnitude, n_bins)
    # beta.append(lgn.weibullMleHist(ax, h)[0])
    beta[0,0,0] = lgn.weibullMleHist(ax, h)[0]

    if IMTYPE == 2:
        magnitude = np.abs(par2[imfovbeta])
        ax, h = lgn.create_hist(magnitude, n_bins)
        # beta.append(lgn.weibullMleHist(ax, h)[0])
        beta[1,0,0] = lgn.weibullMleHist(ax, h)[0]

        magnitude = np.abs(par3[imfovbeta])
        ax, h = lgn.create_hist(magnitude, n_bins)
        # beta.append(lgn.weibullMleHist(ax, h)[0])
        beta[2,0,0] = lgn.weibullMleHist(ax, h)[0]

    # Custom boxes (crops)
    for mask_index, mask in enumerate(crop_masks):
        c_mask, c_mask_peri = get_crop_masks(imfovbeta_crops[mask_index], mask)

        box_center = np.abs(par1[c_mask])
        ax, h = lgn.create_hist(box_center, n_bins)
        beta[0, mask_index+1, 0] = lgn.weibullMleHist(ax, h)[0]
        box_peri = np.abs(par1[c_mask_peri])
        ax, h = lgn.create_hist(box_peri, n_bins)
        beta[0, mask_index+1, 1] = lgn.weibullMleHist(ax, h)[0]

        if IMTYPE == 2:
            box_center = np.abs(par2[c_mask])
            ax, h = lgn.create_hist(box_center, n_bins)
            beta[1, mask_index+1, 0] = lgn.weibullMleHist(ax, h)[0]
            box_peri = np.abs(par2[c_mask_peri])
            ax, h = lgn.create_hist(box_peri, n_bins)
            beta[1, mask_index+1, 1] = lgn.weibullMleHist(ax, h)[0]

            box_center = np.abs(par3[c_mask])
            ax, h = lgn.create_hist(box_center, n_bins)
            beta[2, mask_index+1, 0] = lgn.weibullMleHist(ax, h)[0]
            box_peri = np.abs(par3[c_mask_peri])
            ax, h = lgn.create_hist(box_peri, n_bins)
            beta[2, mask_index+1, 1] = lgn.weibullMleHist(ax, h)[0]

    if verbose:
        print("Compute Weibull parameters gamma")
    magnitude = np.abs(mag1[imfovgamma])
    ax, h = lgn.create_hist(magnitude, n_bins)
    gamma[0,0,0] = lgn.weibullMleHist(ax, h)[1]

    if IMTYPE == 2:
        magnitude = np.abs(mag2[imfovgamma])
        ax, h = lgn.create_hist(magnitude, n_bins)
        gamma[1,0,0] = lgn.weibullMleHist(ax, h)[1]

        magnitude = np.abs(mag3[imfovgamma])
        ax, h = lgn.create_hist(magnitude, n_bins)
        gamma[2,0,0] = lgn.weibullMleHist(ax, h)[1]

    # Custom boxes (crops)
    for mask_index, mask in enumerate(crop_masks):
        c_mask, c_mask_peri = get_crop_masks(imfovgamma_crops[mask_index], mask)

        box_center = np.abs(mag1[c_mask])
        ax, h = lgn.create_hist(box_center, n_bins)
        beta[0, mask_index+1, 0] = lgn.weibullMleHist(ax, h)[1]
        box_peri = np.abs(mag1[c_mask_peri])
        ax, h = lgn.create_hist(box_peri, n_bins)
        beta[0, mask_index+1, 1] = lgn.weibullMleHist(ax, h)[1]

        if IMTYPE == 2:
            box_center = np.abs(mag2[c_mask])
            ax, h = lgn.create_hist(box_center, n_bins)
            beta[1, mask_index+1, 0] = lgn.weibullMleHist(ax, h)[1]
            box_peri = np.abs(mag2[c_mask_peri])
            ax, h = lgn.create_hist(box_peri, n_bins)
            beta[1, mask_index+1, 1] = lgn.weibullMleHist(ax, h)[1]

            box_center = np.abs(mag3[c_mask])
            ax, h = lgn.create_hist(box_center, n_bins)
            beta[2, mask_index+1, 0] = lgn.weibullMleHist(ax, h)[1]
            box_peri = np.abs(mag3[c_mask_peri])
            ax, h = lgn.create_hist(box_peri, n_bins)
            beta[2, mask_index+1, 1] = lgn.weibullMleHist(ax, h)[1]

    return (ce, sc, beta, gamma)