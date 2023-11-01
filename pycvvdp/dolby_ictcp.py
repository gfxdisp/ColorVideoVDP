import numpy as np
import torch
import torch.nn as nn

from pycvvdp.vq_metric import vq_metric

"""
Dolby color metric
Reference: https://professional.dolby.com/siteassets/pdfs/dolby-vision-measuring-perceptual-color-volume-v7.1.pdf
"""
class ictcp(vq_metric):
    def __init__(self, device=None):
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.colorspace = 'XYZ'
        self.XYZ2LMS = torch.tensor(((0.3593, -0.1921, 0.0071),
                                     (0.6976, 1.1005, 0.0748),
                                     (-0.0359, 0.0754, 0.8433)), device=self.device).T
        self.LMS2ICTCP = torch.tensor(((2048, 2048, 0),
                                       (6610, -13613, 7003),
                                       (17933, -17390, -543)), device=self.device)/4096
        self.jnd_scaling = torch.tensor((720, 360, 720), device=self.device).view(1,3,1,1,1)
        # ICTCP = bsxfun(@times, invEOTF(XYZ * XYZ2LMSmat) * LMS2ICTCPmat, [720, 360, 720]);

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

        quality = 0.0
        for ff in range(N_frames):
            # TODO: Batched processing
            T = vid_source.get_test_frame(ff, device=self.device, colorspace=self.colorspace)
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.colorspace)
        
            T_lms_prime = self.invEOTF(self.colorspace_conversion(T, self.XYZ2LMS))
            R_lms_prime = self.invEOTF(self.colorspace_conversion(R, self.XYZ2LMS))
            
            T_ictcp = self.colorspace_conversion(T_lms_prime, self.LMS2ICTCP) 
            R_ictcp = self.colorspace_conversion(R_lms_prime, self.LMS2ICTCP) 
            
            quality += self.delta_itp(T_ictcp, R_ictcp) / N_frames
        return quality, None

    def invEOTF(self, lin):
        return (((3424/4096)+(2413/128)*(lin.clip(min=0)/10000)**(2610/16384)) / \
            (1+(2392/128)*(lin.clip(min=0)/10000)**(2610/16384)))**(2523/32)

    def colorspace_conversion(self, img, M):
        ABC = torch.empty_like(img)  # ABC represents any linear colour space
        # To avoid permute (slow), perform separate dot products
        for cc in range(3):
            ABC[...,cc,:,:,:] = torch.sum(img*(M[cc,:].view(1,3,1,1,1)), dim=-4, keepdim=True)
        return ABC

    """
    Reference: https://kb.portrait.com/help/ictcp-color-difference-metric
    """
    def delta_itp(self, img1, img2):
        return 720 * torch.sqrt((img1[...,0,:,:,:] - img2[...,0,:,:,:])**2 +
                                0.5 * (img1[...,1,:,:,:] - img2[...,1,:,:,:])**2 +
                                (img1[...,2,:,:,:] - img2[...,2,:,:,:])**2).mean()

    def short_name(self):
        return 'Dolby-ICTCP'
