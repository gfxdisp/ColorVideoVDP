import torch

from pycvvdp.utils import PU
from pycvvdp.video_source import *
from pycvvdp.vq_metric import *

from pycvvdp.third_party.ssim import SSIM

def get_luma(img):
    return 0.212656*img[...,0,:,:,:] + 0.715158 * img[...,1,:,:,:] + 0.072186 * img[...,2,:,:,:]

"""
Plain SSIM metric, computed on the luma channel. Operates on display-encoded values. If HDR/linear color is encountered, it will be 
PU21-encoded. The display model is used only for images in linear color spaces. Usage is same as 
the ColorVideoVDP metric (see pytorch_examples).
"""
class ssim_metric(vq_metric):

    def __init__(self, display_name="standard_4k", display_photometry=None, color_space="sRGB", device=None):
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.set_display_model( display_name=display_name, display_photometry=display_photometry )
        self.color_space = color_space # input content color space
        self.ssim = SSIM(channel=1, data_range=1.)


    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    def predict_video_source(self, vid_source, frame_padding="replicate"):

        _, _, N_frames = vid_source.get_video_size()
        
        ssim_index = 0 
        n = 0
        for ff in range(N_frames):
            # colorspace='display_encoded_100nit' will get us display-encoded image, or if the original source is linear, it will apply PU-encoding.
            # If the input is PQ-encoded, it will return a PQ-encoded values (). 
            T = get_luma(vid_source.get_test_frame(ff, device=self.device, colorspace='display_encoded_100nit'))
            R = get_luma(vid_source.get_reference_frame(ff, device=self.device, colorspace='display_encoded_100nit'))

            ssim_index += self.ssim.forward(T, R)
            n += 1
        
        return ssim_index/n, None

    def short_name(self):
        return "SSIM"

    def quality_unit(self):
        return ""

