import torch
import numpy as np

from pycvvdp.utils import deltaE00
from pycvvdp.vq_metric import vq_metric

"""
DE2000 metric. Usage is same as the FovVideoVDP metric (see pytorch_examples).
"""
class de2000(vq_metric):
    def __init__(self, device=None,display_name=None,display_geometry=None,display_photometry=None):
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        # D65 White point
        self.w = (0.9505, 1.0000, 1.0888)
        self.colorspace = 'XYZ'       
        
    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    def predict_video_source(self, vid_source, frame_padding="replicate"):

        # T_vid and R_vid are the tensors of the size (1,3,N,H,W)
        # where:
        # N - the number of frames
        # H - height in pixels
        # W - width in pixels
        # Both images must contain linear absolute luminance values in cd/m^2
        # 
        # We assume the pytorch default NCDHW layout

        _, _, N_frames = vid_source.get_video_size()

        e00 = 0.0
        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.colorspace)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.colorspace)
            
            # XYZ to Lab
            w = self.max_L*self.w
            T_lab = self.xyz_to_lab(T, w)
            R_lab = self.xyz_to_lab(R, w)
            
            # Meancdm of Per-pixel DE2000            
            e00 = e00 + self.e00_fn(T_lab, R_lab) / N_frames
        return e00, None
    
    def xyz_to_lab(self, img, W):
        Lab = torch.empty_like(img)
        Lab[...,0:,:,:] = 116*self.lab_fn(img[...,1,:,:,:]/W[1])-16
        Lab[...,1:,:,:] = 500*(self.lab_fn(img[...,0,:,:,:]/W[0]) - self.lab_fn(img[...,1,:,:,:]/W[1]))
        Lab[...,2:,:,:] = 200*(self.lab_fn(img[...,1,:,:,:]/W[1]) - self.lab_fn(img[...,2,:,:,:]/W[2]))
        return Lab
        
    def lab_fn(self, x):
        y = torch.empty_like(x)
        sigma = (6/29)
        y_1 = x**(1/3)
        y_2 = (x/(3*(sigma**2)))+(4/29)
        condition = torch.less(x, sigma**3)
        y = torch.where(condition, y_2, y_1)
        return y
        
    def e00_fn(self, img1, img2):
        sz = torch.numel(img1[...,0,:,:,:])
        img1_row = torch.cat((torch.reshape(img1[...,0,:,:,:], (1,sz)), torch.reshape(img1[...,1,:,:,:], (1,sz)), torch.reshape(img1[...,2,:,:,:], (1,sz))), 0)
        img2_row = torch.cat((torch.reshape(img2[...,0,:,:,:], (1,sz)), torch.reshape(img2[...,1,:,:,:], (1,sz)), torch.reshape(img2[...,2,:,:,:], (1,sz))), 0)
        e00 = deltaE00(img1_row, img2_row)
        # e00_mean = torch.empty_like(torch.reshape(img1[...,0,:,:,:], (1,sz)))
        # e00_mean = torch.mean(torch.from_numpy(e00).to(e00_mean))
        e00_mean = torch.mean(e00)
        return e00_mean

    def short_name(self):
        return "DE-2000"

    def quality_unit(self):
        return "Delta E2000"

    def get_info_string(self):
        return None

    def set_display_model(self, display_photometry, display_geometry):
        self.max_L = display_photometry.get_peak_luminance()
        self.max_L = np.where( self.max_L < 300, self.max_L, 300)
