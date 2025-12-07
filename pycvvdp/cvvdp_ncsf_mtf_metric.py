from pycvvdp import cvvdp 
from utils.glare_model import GlareModel
from pycvvdp.vq_metric import *


class cvvdp_ncsf_mtf(cvvdp):
    def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, surround=None, mtf_model='hdrvdp'):
        
        if mtf_model != 'hdrvdp':
            config_paths.append(f"ColorVideoVDP/pycvvdp/vvdp_data/cvvdp_ncsf_mtf_{mtf_model}")
        else:
            config_paths.append("ColorVideoVDP/pycvvdp/vvdp_data/cvvdp_ncsf_mtf")

        super().__init__(display_name, display_photometry, display_geometry, config_paths, heatmap, quiet, device, temp_padding, use_checkpoints, dump_channels, gpu_mem)
        
        self.surround = surround
        self.mtf_model = mtf_model

    def predict_video_source(self, vid_source):
        # Apply NCSF MTF before predicting
        vid_source = GlareModel(vid_source, display_geometry=self.display_geometry, surround=self.surround, mtf_model=self.mtf_model)  # remember to add the display geometry
        return super().predict_video_source(vid_source)
    
    def short_name(self):
        surround_str = f"_{self.surround}" if self.surround is not None else ""
        if self.mtf_model != 'hdrvdp':
            return f"cvvdp_ncsf_mtf_{self.mtf_model}{surround_str}"
        return f"cvvdp_ncsf_mtf{surround_str}"

register_metric( cvvdp_ncsf_mtf )
