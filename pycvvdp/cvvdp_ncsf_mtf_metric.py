from pycvvdp import cvvdp 
from utils.glare_model import GlareModel


class cvvdp_ncsf_mtf(cvvdp):
    def predict_video_source(self, vid_source):
        # Apply NCSF MTF before predicting
        vid_source = GlareModel(vid_source, display_geometry=self.display_geometry, surround=None)  # remember to add the display geometry
        return super().predict_video_source(vid_source)
    
    def short_name(self):
        return "cvvdp_ncsf_mtf"

class cvvdp_ncsf_mtf_trained(cvvdp):
    def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None):
        config_paths.append("metric_configs/cvvdp_ncsf_mtf")
        super().__init__(display_name, display_photometry, display_geometry, config_paths, heatmap, quiet, device, temp_padding, use_checkpoints, dump_channels, gpu_mem)
    
    def predict_video_source(self, vid_source):
        # Apply NCSF MTF before predicting
        vid_source = GlareModel(vid_source, display_geometry=self.display_geometry, surround=None)  # remember to add the display geometry
        return super().predict_video_source(vid_source)
    
    def short_name(self):
        return "cvvdp_ncsf_mtf_trained"