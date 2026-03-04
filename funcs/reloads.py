import yaml
import importlib

# import importlib
# from importlib import reload
# import funcs.natspatpred
# import unet_recon.inpainting

# importlib.reload(funcs.natspatpred)
# importlib.reload(unet_recon.inpainting)

# from unet_recon.inpainting import UNet
# from funcs.natspatpred import NatSpatPred, VoxelSieve

# import lgnpy.CEandSC.lgn_statistics
# from lgnpy.CEandSC.lgn_statistics import lgn_statistics, loadmat, LGN

class Reloader():
    def __init__(self):
        pass
        
    def nsp(self):
        import classes.natspatpred

        importlib.reload(classes.natspatpred)
        from classes.natspatpred import NatSpatPred, VoxelSieve

        NSP = NatSpatPred()
        NSP.initialise(verbose=False)
        return NSP

    def lgn(self, config_file: str | None = None):

        import lgnpy.CEandSC.lgn_statistics

        importlib.reload(lgnpy.CEandSC.lgn_statistics)
        from lgnpy.CEandSC.lgn_statistics import lgn_statistics, loadmat, LGN

        if config_file is None:
            config_path = (
                "/home/rfpred/notebooks/alien_nbs/lgnpy/lgnpy/CEandSC/default_config.yml"
            )
        else:
            config_path = (
                f"/home/rfpred/notebooks/alien_nbs/lgnpy/lgnpy/CEandSC/{config_file}"
            )

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

        lgn = LGN(config=config, default_config_path=config_path)
        threshold_lgn = loadmat(
            filepath="/home/rfpred/notebooks/alien_nbs/lgnpy/ThresholdLGN.mat"
        )["ThresholdLGN"]

        return lgn

    def regdat(self):
        import classes.regdata
        importlib.reload(classes.regdata)
        from classes.regdata import RegData
        rd = RegData
        return rd