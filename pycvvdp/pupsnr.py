#from pycvvdp import colorspace
import torch

from pycvvdp.utils import PU
from pycvvdp.video_source import *
from pycvvdp.vq_metric import *


"""
Plain PSNR-RGB metric. Operates on display-encoded values. If HDR/linear colour is encountered, it will be 
PU21-encoded. The display model is used only for images in linear colour spaces. Usage is same as 
the FovVideoVDP metric (see pytorch_examples).
"""
class psnr_rgb(vq_metric):

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
        self.color_space = color_space # input content colour space


    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    def predict_video_source(self, vid_source, frame_padding="replicate"):

        _, _, N_frames = vid_source.get_video_size()

        mse = 0
        for ff in range(N_frames):
            # colorspace='display_encoded_100nit' will get us display-encoded image, or if the original source is linear, it will apply PU-encoding.
            # If the input is PQ-encoded, it will return a PQ-encoded values. 
            T = vid_source.get_test_frame(ff, device=self.device, colorspace='display_encoded_100nit')
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace='display_encoded_100nit')
            mse += torch.mean( (T - R)**2 )

        max_I = 1
        psnr = 20*torch.log10( max_I/torch.sqrt(mse/N_frames) ) 
        
        return psnr, None

    def short_name(self):
        return "PSNR-RGB"

    def quality_unit(self):
        return "dB"


"""
PU21-PSNR-Y metric. Usage is same as the FovVideoVDP metric (see pytorch_examples).
"""
class pu_psnr_y(vq_metric):

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
        self.color_space = color_space # input content colour space

        self.pu = PU()
        self.max_I = self.pu.encode(torch.as_tensor(100)) # 100 nit should correspond to white on an SDR display
        self.metric_colorspace = 'Y' # colour space in which the metric operates

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    def predict_video_source(self, vid_source, frame_padding="replicate"):

        _, _, N_frames = vid_source.get_video_size()

        mse = 0
        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.metric_colorspace)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.metric_colorspace)

            # Apply PU
            T_enc = self.pu.encode(T)
            R_enc = self.pu.encode(R)

            mse += torch.mean( (T_enc - R_enc)**2 )
        
        psnr = 20*torch.log10( self.max_I/torch.sqrt(mse/N_frames) ) 

        return psnr, None
        
    def psnr_fn(self, img1, img2):
        mse = torch.mean( (img1 - img2)**2 )
        return 20*torch.log10( self.pu.peak/torch.sqrt(mse) )
        
    def short_name(self):
        return "PU21-PSNR-Y"

    def quality_unit(self):
        return "dB"


class pu_psnr_rgb2020(pu_psnr_y):
    def __init__(self, display_name="standard_4k", display_photometry=None, color_space="sRGB", device=None):
        super().__init__(display_name=display_name, display_photometry=display_photometry, color_space=color_space, device=device)
        self.metric_colorspace = 'RGB2020'

    def short_name(self):
        return "PU21-PSNR-RGB2020"
    
class pu_psnr_mse_y(pu_psnr_y):
    def __init__(self, device=None):
        super().__init__(device)

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

        mse = 0.0
        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.colorspace)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.colorspace)

            # Apply PU
            T_enc = self.pu.encode(T)
            R_enc = self.pu.encode(R)

            mse = mse + self.mse_fn(T_enc, R_enc) / N_frames
 
        psnr = 20*torch.log10( self.pu.peak/torch.sqrt(mse) )

        return psnr, None

    def mse_fn(self, img1, img2):
        return torch.mean( (img1 - img2)**2 )
  
    def short_name(self):
        return "PU21-PSNR-MSE-Y"
    
class pu_psnr_mse_rgb2020(pu_psnr_mse_y):
    def __init__(self, device=None):
        super().__init__(device)
        self.colorspace = 'RGB2020'
    
    def short_name(self):
        return "PU21-PSNR-MSE-RGB2020"
