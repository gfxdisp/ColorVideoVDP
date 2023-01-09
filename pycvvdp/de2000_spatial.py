import torch
import numpy as np

from pycvvdp.utils import CIE_DeltaE
from pycvvdp.utils import SCIELAB_filter
from pycvvdp.video_source import *
from pycvvdp.vq_metric import *
from pycvvdp.display_model import vvdp_display_photometry, vvdp_display_geometry

"""
Spatial DE2000 metric. Usage is same as the FovVideoVDP metric (see pytorch_examples).
"""
class s_de2000(vq_metric):
    def __init__(self, device=None,display_name=None,display_geometry=None):
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
            
        self.slab = SCIELAB_filter()
        self.de = CIE_DeltaE()
        # D65 White point
        self.w = (0.9505, 1.0000, 1.0888)
        self.colorspace = 'XYZ'       
        
        if display_geometry is None:
            self.display_geometry = vvdp_display_geometry.load(display_name)
        else:
            self.display_geometry = display_geometry

        self.pix_per_deg = self.display_geometry.get_ppd()
        
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

        e_s00 = 0.0
        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.colorspace)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.colorspace)
            
            # XYZ to Opp
            T_opp = self.slab.xyz_to_opp(T)
            R_opp = self.slab.xyz_to_opp(R)
            
            # Spatially filtered opponent colour images
            T_s_opp = self.opp_to_sopp(T_opp, self.pix_per_deg)
            R_s_opp = self.opp_to_sopp(R_opp, self.pix_per_deg)
            
            # S-OPP to S-XYZ
            T_s_xyz = self.slab.opp_to_xyz(T_s_opp)
            R_s_xyz = self.slab.opp_to_xyz(R_s_opp)
            
            # S-XYZ to S-Lab
            T_s_lab = self.xyz_to_lab(T_s_xyz, self.w)
            R_s_lab = self.xyz_to_lab(R_s_xyz, self.w)
            
            # Meancdm of Per-pixel DE2000            
            e_s00 = e_s00 + self.e00_fn(T_s_lab, R_s_lab) / N_frames
        return e00, None
    
    def xyz_to_opp(self, img):
        XYZ_to_opp = (
            (0.2787,0.7218,-0.1066),
            (-0.4488,0.2898,0.0772),
            (0.0860,-0.5900,0.5011) )
        mat = torch.as_tensor( XYZ_to_opp, dtype=img.dtype, device=img.device)
        OPP = torch.empty_like(img)
        for cc in range(3):
            OPP[...,cc,:,:,:] = torch.sum(img*(mat[cc,:].view(1,3,1,1,1)), dim=-4, keepdim=True)
        return OPP
    
    def opp_to_sopp(self, img, ppd):
        S_OPP = torch.empty_like(img)
        [k1, k2, k3] = self.slab.separableFilters(ppd)
        S_OPP[...,0:,:,:] = torch.from_numpy(self.slab.separableConv(torch.squeeze(img[...,0,:,:,:]), k1, np.abs(k1))).to(S_OPP)
        S_OPP[...,1:,:,:] = torch.from_numpy(self.slab.separableConv(torch.squeeze(img[...,1,:,:,:]), k2, np.abs(k2))).to(S_OPP)
        S_OPP[...,2:,:,:] = torch.from_numpy(self.slab.separableConv(torch.squeeze(img[...,2,:,:,:]), k3, np.abs(k3))).to(S_OPP)
        return S_OPP
        
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
        e00 = self.de.deltaE00(img1_row, img2_row)
        e00_mean = torch.empty_like(torch.reshape(img1[...,0,:,:,:], (1,sz)))
        e00_mean = torch.mean(torch.from_numpy(e00).to(e00_mean))
        return 20*torch.log10( e00_mean )

    def short_name(self):
        return "dE 2000"

    def quality_unit(self):
        return "dB"

    def get_info_string(self):
        return None