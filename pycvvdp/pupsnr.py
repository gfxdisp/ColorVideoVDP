import torch

from pycvvdp.utils import PU
from pycvvdp.video_source import *
from pycvvdp.vq_metric import *

"""
PU21-PSNR-Y metric. Usage is same as the FovVideoVDP metric (see pytorch_examples).
"""
class pu_psnr_y(vq_metric):
    def __init__(self, device=None):
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.pu = PU()
        self.colorspace = 'Y'

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    def predict_video_source(self, vid_source, frame_padding="replicate"):

        # T_vid and R_vid are the tensors of the size (1,1,N,H,W)
        # where:
        # N - the number of frames
        # H - height in pixels
        # W - width in pixels
        # Both images must contain linear absolute luminance values in cd/m^2
        # 
        # We assume the pytorch default NCDHW layout

        _, _, N_frames = vid_source.get_video_size()

        psnr = 0.0
        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.colorspace)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.colorspace)

            # Apply PU
            T_enc = self.pu.encode(T)
            R_enc = self.pu.encode(R)

            psnr = psnr + self.psnr_fn(T_enc, R_enc) / N_frames
        return psnr, None

    def psnr_fn(self, img1, img2):
        mse = torch.mean( (img1 - img2)**2 )
        return 20*torch.log10( self.pu.peak/torch.sqrt(mse) )

    def short_name(self):
        return "PU21-PSNR-Y"

    def quality_unit(self):
        return "dB"

class pu_psnr_rgb2020:
    def __init__(self, device=None):
        super().__init__(device)
        self.colorspace = 'RGB2020'

    def short_name(self):
        return "PU21-PSNR-RGB2020"

