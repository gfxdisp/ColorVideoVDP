from abc import abstractmethod
from urllib.parse import ParseResultBytes
from numpy.lib.shape_base import expand_dims
import math
import torch
from torch.utils import checkpoint
from torch.functional import Tensor
from torchvision.transforms import GaussianBlur
import torch.nn.functional as Func
import numpy as np 
import os
import sys
import json
#import argparse
#import time
#import math
import torch.utils.benchmark as torchbench
import logging
from datetime import date

try:
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from matplotlib.colors import Normalize
    has_matplotlib = True
except:
    has_matplotlib = False

from pycvvdp.visualize_diff_map import visualize_diff_map
from pycvvdp.video_source import *

from pycvvdp.vq_metric import *
#from pycvvdp.colorspace import lms2006_to_dkld65

# For debugging only
# from gfxdisp.pfs.pfs_torch import pfs_torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from third_party.cpuinfo import cpuinfo
from pycvvdp.lpyr_dec import lpyr_dec, lpyr_dec_2, weber_contrast_pyr, log_contrast_pyr
from interp import interp1, interp3, interp1dim2

import pycvvdp.utils as utils

#from utils import *
#from fvvdp_test import FovVideoVDP_Testbench

from pycvvdp.display_model import vvdp_display_photometry, vvdp_display_geometry
from pycvvdp.csf import castleCSF


def safe_pow( x:Tensor, p ): 
    #assert (not x.isnan().any()) and (not x.isinf().any()), "Must not be nan"
    #assert torch.all(x>=0), "Must be positive"

    if True: #isinstance( p, Tensor ) and p.requires_grad:
        # If we need a derivative with respect to p, x must not be 0
        epsilon = torch.as_tensor( 0.00001, device=x.device )
        return (x+epsilon) ** p - epsilon**p
    else:
        return x ** p

def pow_neg( x:Tensor, p ): 
    #assert (not x.isnan().any()) and (not x.isinf().any()), "Must not be nan"

    #return torch.tanh(100*x) * (torch.abs(x) ** p)

    min_v = torch.as_tensor( 0.00001, device=x.device )
    return (torch.max(x,min_v) ** p) + (torch.max(-x,min_v) ** p) - min_v**p

"""
ColourVideoVDP metric. Refer to pytorch_examples for examples on how to use this class. 
"""
class cvvdp(vq_metric):
    def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, calibrated_ckpt=None):
        self.quiet = quiet
        self.heatmap = heatmap
        self.temp_padding = temp_padding
        self.use_checkpoints = use_checkpoints # Used for training

        assert heatmap in ["threshold", "supra-threshold", "raw", "none", None], "Unknown heatmap type"            

        self.do_heatmap = (not self.heatmap is None) and (self.heatmap != "none")

        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        self.set_display_model(display_name, display_photometry=display_photometry, display_geometry=display_geometry, config_paths=config_paths)

        self.temp_resample = False  # When True, resample the temporal features to nominal_fps
        self.nominal_fps = 240

        self.load_config(config_paths)
        if calibrated_ckpt is not None:
            self.update_from_checkpoint(calibrated_ckpt)

        # if self.mask_s > 0.0:
        #     self.mask_p = self.mask_q + self.mask_s

        # self.csf_cache              = {}
        # self.csf_cache_dirs         = [
        #                                 "csf_cache",
        #                                 os.path.join(os.path.dirname(__file__), "csf_cache"),
        #                               ]

        # self.omega = torch.tensor([0,5], device=self.device, requires_grad=False)
        # for oo in self.omega:
        #     self.preload_cache(oo, self.csf_sigma)

        self.heatmap_pyr = None

    def load_config( self, config_paths ):

        #parameters_file = os.path.join(os.path.dirname(__file__), "fvvdp_data/fvvdp_parameters.json")
        self.parameters_file = utils.config_files.find( "cvvdp_parameters.json", config_paths )
        logging.debug( f"Loading ColourVideoVDP parameters from '{self.parameters_file}'" )
        parameters = utils.json2dict(self.parameters_file)

        #all common parameters between Matlab and Pytorch, loaded from the .json file
        self.mask_p = torch.as_tensor( parameters['mask_p'], device=self.device )
        self.mask_c = torch.as_tensor( parameters['mask_c'], device=self.device ) # content masking adjustment
        self.pu_dilate = parameters['pu_dilate']
        if self.pu_dilate>0:
            self.pu_blur = GaussianBlur(int(self.pu_dilate*4)+1, self.pu_dilate)
            self.pu_padsize = int(self.pu_dilate*2)
            
        self.beta = torch.as_tensor( parameters['beta'], device=self.device ) # The exponent of the spatial summation (p-norm)
        self.beta_t = torch.as_tensor( parameters['beta_t'], device=self.device ) # The exponent of the summation over time (p-norm)
        self.beta_tch = torch.as_tensor( parameters['beta_tch'], device=self.device ) # The exponent of the summation over temporal channels (p-norm)
        self.beta_sch = torch.as_tensor( parameters['beta_sch'], device=self.device ) # The exponent of the summation over spatial channels (p-norm)
        self.csf_sigma = torch.as_tensor( parameters['csf_sigma'], device=self.device )
        self.sensitivity_correction = torch.as_tensor( parameters['sensitivity_correction'], device=self.device ) # Correct CSF values in dB. Negative values make the metric less sensitive.
        self.masking_model = parameters['masking_model']
        if "texture" in self.masking_model:
            tex_blur_sigma = 8
            self.tex_blur = GaussianBlur(int(tex_blur_sigma*4)+1, tex_blur_sigma)
            self.tex_pad_size = int(tex_blur_sigma*2)

        self.csf = parameters['csf']
        self.local_adapt = parameters['local_adapt'] # Local adaptation: 'simple' or or 'gpyr'
        self.contrast = parameters['contrast']  # One of: 'weber_g0_ref', 'weber_g1_ref', 'weber_g1', 'log'
        self.jod_a = torch.as_tensor( parameters['jod_a'], device=self.device )
        self.jod_exp = torch.as_tensor( parameters['jod_exp'], device=self.device )

        if 'ce_g' in parameters:
            self.ce_g = torch.as_tensor( parameters['ce_g'], device=self.device )

        if 'k_c' in parameters:
            self.k_c = torch.as_tensor( parameters['k_c'], device=self.device )

        if 'temp_filter' in parameters:
            self.temp_filter = parameters['temp_filter']
        else:
            self.temp_filter = "default"

        if 'std_pool' in parameters:
            self.std_pool = parameters['std_pool']
            self.std_w = torch.as_tensor( parameters['std_w'], device=self.device )
        else:
            self.std_pool = "ts"

        if 'mask_q' in parameters:
            self.mask_q = torch.as_tensor( parameters['mask_q'], device=self.device )
        else:
            self.mask_q_sust = torch.as_tensor( parameters['mask_q_sust'], device=self.device )
            self.mask_q_trans = torch.as_tensor( parameters['mask_q_trans'], device=self.device )
        self.filter_len = torch.as_tensor( parameters['filter_len'], device=self.device )

        self.do_xchannel_masking = True if parameters['xchannel_masking'] == "on" else False
        self.xcm_weights = torch.as_tensor( parameters['xcm_weights'], device=self.device, dtype=torch.float32 ) 
        
        self.image_int = torch.as_tensor( parameters['image_int'], device=self.device )

        if 'ch_chrom_w' in parameters:
            self.ch_chrom_w = torch.as_tensor( parameters['ch_chrom_w'], device=self.device ) # Chromatic channels (rg, vy) weight
            self.ch_trans_w = torch.as_tensor( parameters['ch_trans_w'], device=self.device ) # Transient channel weight
        else:
            # Depreciated - will be removed later
            self.ch_weights = torch.as_tensor( parameters['ch_weights'], device=self.device ) # Per-channel weight, Y-sust, rg, vy, Y-trans
        self.sigma_tf = torch.as_tensor( parameters['sigma_tf'], device=self.device ) # Temporal filter params, per-channel: Y-sust, rg, vy, Y-trans
        self.beta_tf = torch.as_tensor( parameters['beta_tf'], device=self.device ) # Temporal filter params, per-channel: Y-sust, rg, vy, Y-trans
        self.baseband_weight = torch.as_tensor( parameters['baseband_weight'], device=self.device )
        if self.baseband_weight.numel()<4:
            self.baseband_weight = self.baseband_weight.repeat(4)
        self.dclamp_type = parameters['dclamp_type']  # clamping mode: soft or hard
        self.d_max = torch.as_tensor( parameters['d_max'], device=self.device ) # Clamping of difference values
        self.version = parameters['version']

        self.do_Bloch_int = True if parameters['Bloch_int'] == "on" else False
        self.bfilt_duration = parameters['bfilt_duration']

        self.omega = [0, 5]

        self.csf = castleCSF(csf_version=self.csf, device=self.device, config_paths=config_paths)

        # Mask to block selected channels, used in the ablation stdies [Ysust, RB, YV, Ytrans]
        self.block_channels = torch.as_tensor( parameters['block_channels'], device=self.device, dtype=torch.bool ) if 'block_channels' in parameters else None
        
        # other parameters
        self.debug = False

    def update_from_checkpoint(self, ckpt):
        assert os.path.isfile(ckpt), f'Calibrated PyTorch checkpoint not found at: {ckpt}'
        # Read relevant parameters from state_dict
        prefix = 'params.'
        
        if torch.cuda.is_available():
            for key, value in torch.load(ckpt)['state_dict'].items():
                if key.startswith(prefix):
                    setattr(self, key[len(prefix):], value.to(self.device))
        else:
            for key, value in torch.load(ckpt, map_location=torch.device('cpu'))['state_dict'].items():
                if key.startswith(prefix):
                    setattr(self, key[len(prefix):], value.to(self.device))
        
        
    def set_display_model(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[]):
        if display_photometry is None:
            self.display_photometry = vvdp_display_photometry.load(display_name, config_paths)
            self.display_name = display_name
        else:
            self.display_photometry = display_photometry
            self.display_name = "unspecified"
        
        if display_geometry is None:
            self.display_geometry = vvdp_display_geometry.load(display_name, config_paths)
        else:
            self.display_geometry = display_geometry

        self.pix_per_deg = self.display_geometry.get_ppd()
        self.imgaussfilt = utils.ImGaussFilt(0.5 * self.pix_per_deg, self.device)
        self.lpyr = None

    '''
    Predict image/video quality using FovVideoVDP.

    test_cont and reference_cont can be either numpy arrays or PyTorch tensors with images or video frames. 
        Depending on the display model (display_photometry), the pixel values should be either display encoded, or absolute linear.
        The two supported datatypes are float16 and uint8.
    dim_order - a string with the order of dimensions of test_cont and reference_cont. The individual characters denote
        B - batch
        C - colour channel
        F - frame
        H - height
        W - width
        Examples: "HW" - gray-scale image (column-major pixel order); "HWC" - colour image; "FCHW" - colour video
        The default order is "BCFHW". The processing can be a bit faster if data is provided in that order. 
    frame_padding - the metric requires at least 250ms of video for temporal processing. Because no previous frames exist in the
        first 250ms of video, the metric must pad those first frames. This options specifies the type of padding to use:
          'replicate' - replicate the first frame
          'circular'  - tile the video in the front, so that the last frame is used for frame 0.
          'pingpong'  - the video frames are mirrored so that frames -1, -2, ... correspond to frames 0, 1, ...
    '''
    def predict(self, test_cont, reference_cont, dim_order="BCFHW", frames_per_second=0):

        test_vs = video_source_array( test_cont, reference_cont, frames_per_second, dim_order=dim_order, display_photometry=self.display_photometry )

        return self.predict_video_source(test_vs)

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    def predict_video_source(self, vid_source):
        # We assume the pytorch default NCDHW layout

        vid_sz = vid_source.get_video_size() # H, W, F
        height, width, N_frames = vid_sz

        # 'medium' is a bit slower than 'high' on 3090
        # torch.set_float32_matmul_precision('medium')

        if self.lpyr is None or self.lpyr.W!=width or self.lpyr.H!=height:
            if self.contrast.startswith("weber"):
                self.lpyr = weber_contrast_pyr(width, height, self.pix_per_deg, self.device, contrast=self.contrast)
            elif self.contrast.startswith("log"):
                self.lpyr = log_contrast_pyr(width, height, self.pix_per_deg, self.device, contrast=self.contrast)
            else:
                raise RuntimeError( f"Unknown contrast {self.contrast}" )

            if self.do_heatmap:
                self.heatmap_pyr = lpyr_dec_2(width, height, self.pix_per_deg, self.device)

        #assert self.W == R_vid.shape[-1] and self.H == R_vid.shape[-2]
        #assert len(R_vid.shape)==5

        is_image = (N_frames==1)  # Can run faster on images

        if is_image:
            temp_ch = 1  # How many temporal channels
        else:
            temp_ch = 2
            self.F, omega_tmp = self.get_temporal_filters(vid_source.get_frames_per_second())
            self.filter_len = torch.numel(self.F[0])

        all_ch = 2+temp_ch

        if self.do_heatmap:
            dmap_channels = 1 if self.heatmap == "raw" else 3
            heatmap = torch.zeros([1,dmap_channels,N_frames,height,width], dtype=torch.float16, device=torch.device('cpu')) # Store heatmap in the CPU memory
        else:
            heatmap = None

        sw_buf = [None, None]
        Q_per_ch = None

        fl = self.filter_len

        if self.device.type == 'cuda' and torch.cuda.is_available() and not is_image:
            # GPU utilization is better if we process many frames, but it requires more GPU memory

            # Determine how much memory we have
            total = torch.cuda.get_device_properties(self.device).total_memory
            allocated = torch.cuda.memory_allocated(self.device)
            mem_avail = total-allocated-1000000000  # Total available - 1G

            # Estimate how much we need for processing (may be inaccurate - to be improved)
            pix_cnt = width*height
            # sw_buf            
            mem_const = pix_cnt*4*3*2*(fl-1)
            # sw_buf + R + B_bands + L_bkg_pyr + (T_f + R_f) + S
            if self.debug: 
                self.mem_allocated_start = allocated
                self.mem_allocated_peak = 0

            #mem_per_frame = pix_cnt*4*3*2 + pix_cnt*4*all_ch*2 + int(pix_cnt*4*all_ch*2*1.33) + int(pix_cnt*4*2*1.33) + int(pix_cnt*4*2*1.33) + int(pix_cnt*4*1.33) 
            if self.use_checkpoints:           
                # More memory required when training. TODO: better way to detect when running with require_grad
                mem_per_frame = pix_cnt*2000   # Estimated memory required per frame
            else:
                mem_per_frame = pix_cnt*450   # Estimated memory required per frame

            max_frames = int((mem_avail-mem_const)/mem_per_frame) # how many frames can we fit into memory

            block_N_frames = max(1, min(max_frames,N_frames))  # Process so many frames in one pass 
            if self.debug: logging.debug( f"Processing {block_N_frames} frames in a batch." )
        else:
            block_N_frames = 1

        if self.contrast=="log":
            met_colorspace='logLMS_DKLd65'
        else:
            met_colorspace='DKLd65' # This metric uses DKL colourspaxce with d65 whitepoint

        for ff in range(0, N_frames, block_N_frames):
            cur_block_N_frames = min(block_N_frames,N_frames-ff) # How many frames in this block?

            if is_image:                
                R = torch.empty((1, 6, 1, height, width), device=self.device)
                R[:,0::2, :, :, :] = vid_source.get_test_frame(0, device=self.device, colorspace=met_colorspace)
                R[:,1::2, :, :, :] = vid_source.get_reference_frame(0, device=self.device, colorspace=met_colorspace)

            else: # This is video
                #if self.debug: print("Frame %d:\n----" % ff)

                if ff == 0: # First frame
                    sw_buf[0] = torch.zeros((1,3,fl+block_N_frames-1,height,width), device=self.device, dtype=torch.float32) # TODO: switch to float16
                    sw_buf[1] = torch.zeros((1,3,fl+block_N_frames-1,height,width), device=self.device, dtype=torch.float32)

                    if self.temp_padding == "replicate":
                        for fi in range(cur_block_N_frames):
                            ind = fl+fi-1
                            sw_buf[0][:,:,ind:ind+1,:,:] = vid_source.get_test_frame(ff+fi, device=self.device, colorspace=met_colorspace)
                            sw_buf[1][:,:,ind:ind+1,:,:] = vid_source.get_reference_frame(ff+fi, device=self.device, colorspace=met_colorspace)

                        ind = fl-1
                        sw_buf[0][:,:,0:-cur_block_N_frames,:,:] = sw_buf[0][:,:,ind:ind+1,:,:] # Replicate the first frame
                        sw_buf[1][:,:,0:-cur_block_N_frames,:,:] = sw_buf[1][:,:,ind:ind+1,:,:] # Replicate the first frame

                    # elif self.temp_padding == "circular":
                    #     sw_buf[0] = torch.zeros([1, 1, fl, height, width], device=self.device)
                    #     sw_buf[1] = torch.zeros([1, 1, fl, height, width], device=self.device)
                    #     for kk in range(fl):
                    #         fidx = (N_frames - 1 - fl + kk) % N_frames
                    #         sw_buf[0][:,:,kk,...] = vid_source.get_test_frame(fidx, device=self.device)
                    #         sw_buf[1][:,:,kk,...] = vid_source.get_reference_frame(fidx, device=self.device)
                    # elif self.temp_padding == "pingpong":
                    #     sw_buf[0] = torch.zeros([1, 1, fl, height, width], device=self.device)
                    #     sw_buf[1] = torch.zeros([1, 1, fl, height, width], device=self.device)

                    #     pingpong = list(range(0,N_frames)) + list(range(N_frames-2,0,-1))
                    #     indices = []
                    #     while(len(indices) < (fl-1)):
                    #         indices = indices + pingpong
                    #     indices = indices[-(fl-1):] + [0]

                    #     for kk in range(fl):
                    #         fidx = indices[kk]
                    #         sw_buf[0][:,:,kk,...] = vid_source.get_test_frame(fidx,device=self.device)
                    #         sw_buf[1][:,:,kk,...] = vid_source.get_reference_frame(fidx,device=self.device)
                    else:
                        raise RuntimeError( 'Unknown padding method "{}"'.format(self.temp_padding) )
                else:
                    # scroll the sliding window buffers
                    # Tensor splicing leads to strange errors with videos; switching to torch.roll()
                    # sw_buf[0][:,:,0:-cur_block_N_frames,:,:] = sw_buf[0][:,:,cur_block_N_frames:,:,:]
                    # sw_buf[1][:,:,0:-cur_block_N_frames,:,:] = sw_buf[1][:,:,cur_block_N_frames:,:,:]
                    sw_buf[0] = torch.roll(sw_buf[0], shifts=-cur_block_N_frames, dims=2)
                    sw_buf[1] = torch.roll(sw_buf[1], shifts=-cur_block_N_frames, dims=2)

                    for fi in range(cur_block_N_frames):
                        ind=fl+fi-1
                        sw_buf[0][:,:,ind:ind+1,:,:] = vid_source.get_test_frame(ff+fi, device=self.device, colorspace=met_colorspace)
                        sw_buf[1][:,:,ind:ind+1,:,:] = vid_source.get_reference_frame(ff+fi, device=self.device, colorspace=met_colorspace)

                # Order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y
                # Images do not have the two last channels
                R = torch.zeros((1, 8, cur_block_N_frames, height, width), device=self.device)

                for cc in range(all_ch): # Iterate over chromatic and temporal channels
                    # 1D filter over time (over frames)
                    corr_filter = self.F[cc].flip(0).view([1,1,self.F[cc].shape[0],1,1]) 
                    sw_ch = 0 if cc==3 else cc # colour channel in the sliding window
                    for fi in range(cur_block_N_frames):
                        R[:,cc*2+0, fi, :, :] = (sw_buf[0][:, sw_ch, fi:(fl+fi), :, :] * corr_filter).sum(dim=-3,keepdim=True) # Test
                        R[:,cc*2+1, fi, :, :] = (sw_buf[1][:, sw_ch, fi:(fl+fi), :, :] * corr_filter).sum(dim=-3,keepdim=True) # Reference

            if self.use_checkpoints:
                # Used for training
                Q_per_ch_block, heatmap_block = checkpoint.checkpoint(self.process_block_of_frames, R, vid_sz, temp_ch, self.lpyr, is_image, use_reentrant=False)
            else:
                Q_per_ch_block, heatmap_block = self.process_block_of_frames(R, vid_sz, temp_ch, self.lpyr, is_image)

            if Q_per_ch is None:
                Q_per_ch = torch.zeros((Q_per_ch_block.shape[0], N_frames, Q_per_ch_block.shape[2]), device=self.device)
            
            ff_end = ff+Q_per_ch_block.shape[1]
            Q_per_ch[:,ff:ff_end,:] = Q_per_ch_block  

            if self.do_heatmap:
                if self.heatmap == "raw":
                    heatmap[:,:,ff:ff_end,...] = heatmap_block.detach().type(torch.float16).cpu()
                else:
                    ref_frame = R[:,0, :, :, :]
                    heatmap[:,:,ff:ff_end,...] = visualize_diff_map(heatmap_block, context_image=ref_frame, colormap_type=self.heatmap, use_cpu=self.device.type == 'mps').detach().type(torch.float16).cpu()

        if self.temp_resample:
            t_end = N_frames/vid_source.get_frames_per_second() # Video duration in s
            t_org = torch.linspace( 0., t_end, N_frames, device=self.device )
            N_frames_resampled = math.ceil(t_end * self.nominal_fps)
            t_resampled = torch.linspace( 0., N_frames_resampled/self.nominal_fps, N_frames_resampled, device=self.device )
            Q_per_ch = interp1dim2(t_org, Q_per_ch, t_resampled)
            N_frames = N_frames_resampled
            fps = self.nominal_fps
        else:
            fps = vid_source.get_frames_per_second()


        rho_band = self.lpyr.get_freqs()
        Q_jod = self.do_pooling_and_jods(Q_per_ch, rho_band[-1], fps)

        stats = {}
        stats['Q_per_ch'] = Q_per_ch.detach().cpu().numpy() # the quality per channel and per frame
        stats['rho_band'] = rho_band # The spatial frequency per band in cpd
        stats['frames_per_second'] = fps
        stats['width'] = width
        stats['height'] = height
        stats['N_frames'] = N_frames

        if self.do_heatmap:            
            stats['heatmap'] = heatmap

        if self.debug and hasattr(self,"mem_allocated_peak"): 
            logging.debug( f"Allocated at start: {self.mem_allocated_start/1e9} GB" )
            logging.debug( f"Max allocated: {self.mem_allocated_peak/1e9} GB" )
            logging.debug( f"Resolution: {width}x{height} = {width*height/1e6} Mpixels" )
            pix_cnt = width*height
            # sw_buf            
            mem_const = pix_cnt*4*3*2*(fl-1)
            per_pixel = (self.mem_allocated_peak-self.mem_allocated_start-mem_const)/(pix_cnt*block_N_frames)
            logging.debug( f"Memory used per pixel: {per_pixel} B" )

        return (Q_jod.squeeze(), stats)

    def get_ch_weights(self, no_channels):
        if hasattr(self, 'ch_chrom_w'):
            per_ch_w_all = torch.stack( [torch.as_tensor(1., device=self.ch_chrom_w.device), self.ch_chrom_w, self.ch_chrom_w, self.ch_trans_w] )
        else:
            # Depreciated - will be removed later
            per_ch_w_all = self.ch_weights
            
        # Weights for the channels: sustained, RG, YV, [transient]
        per_ch_w = per_ch_w_all[0:no_channels].view(-1,1,1)
        return per_ch_w


    # Perform pooling with per-band weights and map to JODs
    def do_pooling_and_jods(self, Q_per_ch, base_rho_band, fps):
        # Q_per_ch[channel,frame,sp_band]

        no_channels = Q_per_ch.shape[0]
        no_frames = Q_per_ch.shape[1]
        no_bands = Q_per_ch.shape[2]

        per_ch_w = self.get_ch_weights( no_channels )

        # if no_frames>1: # If video
        #     per_ch_w = self.ch_weights[0:no_channels].view(-1,1,1)
        #     #torch.stack( (torch.ones(1, device=self.device), torch.as_tensor(self.w_transient, device=self.device)[None] ), dim=1)[:,:,None]
        # else: # If image
        #     per_ch_w = 1

        # Weights for the spatial bands
        per_sband_w = torch.ones( (no_channels,1,no_bands), dtype=torch.float32, device=self.device)
        per_sband_w[:,0,-1] = self.baseband_weight[0:no_channels]

        #per_sband_w = torch.exp(interp1( self.quality_band_freq_log, self.quality_band_w_log, torch.log(torch.as_tensor(rho_band, device=self.device)) ))[:,None,None]

        Q_sc = self.lp_norm(Q_per_ch*per_ch_w*per_sband_w, self.beta_sch, dim=2, normalize=False)  # Sum across spatial channels

        is_image = (no_frames==1)
        t_int = self.image_int if is_image else 1.0 # Integration correction for images

        if not is_image and self.do_Bloch_int:
            bfilt_len = int(math.ceil(self.bfilt_duration * fps))
            Q_in = Q_sc.permute(0,2,1)
            B_filt = torch.ones( (1,1,bfilt_len), dtype=torch.float32, device=Q_in.device )/float(bfilt_len)
            Q_bi = torch.nn.functional.conv1d(Q_in,B_filt, padding="valid")
            if not self.block_channels is None:
                Q_tc = self.lp_norm(Q_bi[self.block_channels,...], self.beta_tch, dim=0, normalize=False)  # Sum across temporal and chromatic channels                
            else:
                Q_tc = self.lp_norm(Q_bi,     self.beta_tch, dim=0, normalize=False)  # Sum across temporal and chromatic channels
            Q = self.lp_norm(Q_tc,     self.beta_t,   dim=2, normalize=True)   # Sum across frames
        else:
            if not self.block_channels is None:
                Q_tc = self.lp_norm(Q_sc[self.block_channels[0:no_channels],...], self.beta_tch, dim=0, normalize=False)  # Sum across temporal and chromatic channels                
            else:
                Q_tc = self.lp_norm(Q_sc,     self.beta_tch, dim=0, normalize=False)  # Sum across temporal and chromatic channels

            if is_image:
                Q = Q_tc * t_int
            else:
                if self.std_pool[0]=='T':
                    std_wt = 2**self.std_w[0]
                    Q = self.lp_norm(Q_tc,     self.beta_t,   dim=1, normalize=True) + std_wt*torch.std(Q_tc, dim=1)   # Sum across frames
                else:
#                    assert torch.all(Q_tc>=0) and not Q_tc.isnan().any(), "wrong values"
                    Q = self.lp_norm(Q_tc,     self.beta_t,   dim=1, normalize=True)   # Sum across frames

        Q = Q.squeeze()

        Q_JOD = self.met2jod(Q)            
        return Q_JOD

        # sign = lambda x: (1, -1)[x<0]
        # beta_jod = 10.0**self.log_jod_exp
        # Q_jod = sign(self.jod_a) * ((abs(self.jod_a)**(1.0/beta_jod))* Q)**beta_jod + 10.0 # This one can help with very large numbers
        # return Q_jod.squeeze()

    # Convert contrast differences to JODs
    def met2jod(self, Q):

        # We could use 
        # Q_JOD = 10. - self.jod_a * Q**self.jod_exp
        # but it does not differentiate well near Q=0

        Q_t = 0.1
        jod_a_p = self.jod_a * (Q_t**(self.jod_exp-1.))

        Q_JOD = torch.empty_like(Q)
        Q_JOD[Q<=Q_t] = 10. - jod_a_p * Q[Q<=Q_t];
        Q_JOD[Q>Q_t] = 10. - self.jod_a * (Q[Q>Q_t]**self.jod_exp);
        return Q_JOD

    def process_block_of_frames(self, R, vid_sz, temp_ch, lpyr, is_image):
        # R[channels,frames,width,height]
        #height, width, N_frames = vid_sz
        all_ch = 2+temp_ch

        #torch.autograd.set_detect_anomaly(True)

        # if self.contrast=="log":
        #     R = lms2006_to_dkld65( torch.log10(R.clip(min=1e-5)) )

        # Perform Laplacian pyramid decomposition
        B_bands, L_bkg_pyr = lpyr.decompose(R[0,...])

        if self.debug: assert len(B_bands) == lpyr.get_band_count()

        # if self.do_heatmap:
        #     Dmap_pyr_bands, Dmap_pyr_gbands = self.heatmap_pyr.decompose( torch.zeros([1,1,height,width], dtype=torch.float, device=self.device))

        # L_bkg_bb = [None for i in range(lpyr.get_band_count()-1)]

        rho_band = lpyr.get_freqs()
        rho_band[lpyr.get_band_count()-1] = 0.1 # Baseband

        Q_per_ch_block = None
        block_N_frames = R.shape[-3] 

        for bb in range(lpyr.get_band_count()):  # For each spatial frequency band

            is_baseband = (bb==(lpyr.get_band_count()-1))

            B_bb = lpyr.get_band(B_bands, bb) 
            T_f = B_bb[0::2,...] # Test
            R_f = B_bb[1::2,...] # Reference

            logL_bkg = lpyr.get_gband(L_bkg_pyr, bb)

            # Compute CSF
            rho = rho_band[bb] # Spatial frequency in cpd
            ch_height, ch_width = logL_bkg.shape[-2], logL_bkg.shape[-1]
            S = torch.empty((all_ch,block_N_frames,ch_height,ch_width), device=self.device)
            for cc in range(all_ch):
                tch = 0 if cc<3 else 1  # Sustained or transient
                cch = cc if cc<3 else 0 # Y, rg, yv
                # The sensitivity is always extracted for the reference frame
                S[cc,:,:,:] = self.csf.sensitivity(rho, self.omega[tch], logL_bkg[...,1,:,:,:], cch, self.csf_sigma) * 10.0**(self.sensitivity_correction/20.0)

            if is_baseband:
                D = (torch.abs(T_f-R_f) * S)
            else:
                # dimensions: [channel,frame,height,width]
                D = self.apply_masking_model(T_f, R_f, S)

            if Q_per_ch_block is None:
                Q_per_ch_block = torch.empty((all_ch, block_N_frames, lpyr.get_band_count()), device=self.device)

            #assert (not D.isnan().any()) and (not D.isinf().any()) and (D>=0).all(), "Must not be nan and must be positive"

            Q_per_ch_block[:,:,bb] = self.lp_norm(D, self.beta, dim=(-2,-1), normalize=True, keepdim=False) # Pool across all pixels (spatial pooling)

            if self.std_pool[1]=='S':
                std_ws = 2**self.std_w[1]
                Q_per_ch_block[:,:,bb] += std_ws*torch.std(D, dim=(-2,-1))

            if self.do_heatmap:

                # We need to reduce the differences across the channels using the right weights
                # Weights for the channels: sustained, RG, YV, [transient]
                t_int = self.image_int if is_image else 1.0
                per_ch_w = self.get_ch_weights( all_ch ).view(-1,1,1,1) * t_int
                D_chr = self.lp_norm(D*per_ch_w, self.beta_tch, dim=-4, normalize=False)  # Sum across temporal and chromatic channels
                self.heatmap_pyr.set_lband(bb, D_chr)

        if self.do_heatmap:
            heatmap_block = 1.-(self.met2jod( self.heatmap_pyr.reconstruct() )/10.)
        else:
            heatmap_block = None

        return Q_per_ch_block, heatmap_block

    def mask_pool(self, C):
        # Cross-channel masking
        num_ch = C.shape[0]
        if self.do_xchannel_masking:
            M = torch.empty_like(C)
            xcm_weights = torch.reshape( (2**self.xcm_weights), (4,4,1,1,1) )[:num_ch,...]
            for cc in range(num_ch): # for each channel: Sust, RG, VY, Trans
                M[cc,...] = torch.sum( C * xcm_weights[:,cc], dim=0, keepdim=True )
        else:
            cm_weights = torch.reshape( (2**self.xcm_weights), (4,1,1,1) )[:num_ch,...]
            M = C * cm_weights
        return M

    def ce_overconstancy(self, C, S):
        num_ch = C.shape[0]
        zero_tens = torch.as_tensor(0., device=C.device)
        C_t = torch.minimum( 1/S, torch.as_tensor(1.99, device=C.device) )
        p_t = 0.7
        gain = torch.reshape( torch.as_tensor( [10., 14., 2.1, 10.], device=C.device), (4, 1, 1, 1) )[:num_ch,...]
        C_p = torch.maximum( pow_neg((C - C_t)/(2.0-C_t), p_t)*gain + 1.0, zero_tens )
        return C_p


    def transd_overconstancy(self, C, S):
        num_ch = C.shape[0]
        zero_tens = torch.as_tensor(0., device=C.device)
        C_t = torch.minimum( 1/S, torch.as_tensor(1.99, device=C.device) )
        p_t = 0.7
        gain = torch.reshape( torch.as_tensor( [10., 14., 2.1, 10.], device=C.device), (4, 1, 1, 1) )[:num_ch,...]
        C_p = torch.maximum( pow_neg((C - C_t)/(2.0-C_t), p_t)*gain + 1.0, zero_tens )

        M = self.mask_pool(torch.abs(C_p))

        p = self.mask_p
        q = self.mask_q[0:num_ch].view(num_ch,1,1,1)

        #assert torch.all(M>=0), "M must be positive"
        #assert torch.all(C_p>=0), "C_p must be positive"

        D = 2 * pow_neg(C_p, p) / (1 + M**q)

        #assert not D.isnan().any(), "Must not be nan"

        return D

    def cm_transd(self, C_p):
        num_ch = C_p.shape[0]

        p = self.mask_p
        q = self.mask_q[0:num_ch].view(num_ch,1,1,1)

        M = self.phase_uncertainty(self.mask_pool(safe_pow(torch.abs(C_p),q)))

        D_max = 10**self.d_max

        return D_max * pow_neg( C_p, p ) / (0.2 + M)

    # a differentiable sign function
    def diff_sign(self, x):
        if x.requires_grad:
            return torch.tanh(10000.0 * x)
        else:
            return torch.sign(x)

    def apply_masking_model(self, T, R, S):
        # T - test contrast tensor T[channel,frame,width,height]
        # R - reference contrast tensor
        # S - sensitivity

        if self.masking_model in [ "mult-none", "add-transducer", "mult-transducer", "add-mutual", "mult-mutual", "mult-mutual-old", "add-similarity", "mult-similarity", "mult-transducer-texture", "add-transducer-texture" ]:
            num_ch = T.shape[0]
            if self.masking_model.startswith( "add" ):
                zero_tens = torch.as_tensor(0., device=T.device)
                ch_gain = self.ce_g * torch.reshape( torch.as_tensor( [1, 1.7, 0.237, 1.], device=T.device), (4, 1, 1, 1) )[:num_ch,...] 
                C_t = 1/S
                T_p = self.diff_sign(T) * torch.maximum( (torch.abs(T)-C_t)*ch_gain + 1, zero_tens )
                R_p = self.diff_sign(R) * torch.maximum( (torch.abs(R)-C_t)*ch_gain + 1, zero_tens )
            else:
                if self.masking_model.endswith( "mutual-old" ):
                    T_p = T * S
                    R_p = R * S
                else:
                    ch_gain = torch.reshape( torch.as_tensor( [1, 1.45, 1, 1.], device=T.device), (4, 1, 1, 1) )[:num_ch,...] 
                    T_p = T * S * ch_gain
                    R_p = R * S * ch_gain

            if self.masking_model.endswith( "none" ):
                D = self.clamp_diffs(torch.abs(T_p-R_p))
            elif self.masking_model.endswith( "transducer" ):
                D = torch.abs(self.cm_transd(T_p)-self.cm_transd(R_p))                
            elif self.masking_model.endswith( "mutual" ):

                M_mm = self.phase_uncertainty(torch.min( torch.abs(T_p), torch.abs(R_p) ))
                p = self.mask_p
                q = self.mask_q[0:num_ch].view(num_ch,1,1,1)

                M = self.mask_pool(safe_pow(torch.abs(M_mm),q))

                #D_band = safe_pow(torch.abs(T_p - R_p),p)
                # k_c = self.k_c
                # D_clamped = k_c*D_band / (k_c + D_band)
                #D = D_clamped / (1 + M)
                D_u = safe_pow(torch.abs(T_p - R_p),p) / (1 + M)
                D = self.clamp_diffs( D_u )

            elif self.masking_model.endswith( "mutual-old" ):

                M_mm = self.phase_uncertainty(torch.min( torch.abs(T_p), torch.abs(R_p) ))
                p = self.mask_p
                q = self.mask_q[0:num_ch].view(num_ch,1,1,1)

                M = self.mask_pool(torch.abs(M_mm))

                D_band = safe_pow(torch.abs(T_p - R_p),p)
                D_m = D_band / (1 + safe_pow(M,q))

                #D = self.clamp_diffs( D_m )
                k_c = self.k_c                
                D = k_c*D_m / (k_c + D_m)

            elif self.masking_model.endswith( "transducer-texture" ):

                if T_p.shape[-2] <= self.tex_pad_size or T_p.shape[-1] <= self.tex_pad_size:
                    D = torch.abs(self.cm_transd(T_p)-self.cm_transd(R_p))
                else:
                    T_t = self.cm_transd(T_p)
                    R_t = self.cm_transd(R_p)

                    mu_T = self.tex_blur.forward(T_t)
                    mu_R = self.tex_blur.forward(R_t)

                    mu_T_sq = mu_T * mu_T
                    mu_R_sq = mu_R * mu_R
                    #mu_TR = mu_T * mu_R

                    sigma_T_sq = (self.tex_blur.forward(T_t * T_t) - mu_T_sq).clamp(min=0.)
                    sigma_R_sq = (self.tex_blur.forward(R_t * R_t) - mu_R_sq).clamp(min=0.)
                    #sigma_TR = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

                    #cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
                    #ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

                    D = torch.abs(mu_T-mu_R) + torch.abs(sigma_T_sq.sqrt()-sigma_R_sq.sqrt())

            else: # similarity
                T_p_m = self.phase_uncertainty(self.mask_pool(torch.abs(T_p)))
                R_p_m = self.phase_uncertainty(self.mask_pool(torch.abs(R_p)))
    
                D_max = 10**self.d_max
                epsilon = D_max-1

                D = D_max - D_max*(2*torch.abs(T_p)*torch.abs(R_p)+epsilon)/(T_p_m*T_p_m + R_p_m*R_p_m + epsilon)

            assert not (D.isnan().any() or D.isinf().any()), "Must not be nan"

        elif self.masking_model in ["smooth_clamp_cont", "min_mutual_masking_perc_norm2", "fvvdp_ch_gain"]:

            if self.masking_model == "fvvdp_ch_gain":
                num_ch = T.shape[0]
                ch_gain = torch.reshape( torch.as_tensor( [1, 1.45, 1, 1.], device=T.device), (4, 1, 1, 1) )[:num_ch,...] 
                #print( f"max T[0] = {T[0,...].max()}; \tT[1] = {T[1,...].max()/0.610649}" )
                #print( f"mean S[0] = {S[0,...].mean()}; \tS[1] = {S[1,...].mean()}; \tS[2] = {S[2,...].mean()}" )
                T = T*S*ch_gain
                R = R*S*ch_gain
            else:
                T = T*S
                R = R*S

            M_pu = self.phase_uncertainty( torch.min( torch.abs(T), torch.abs(R) ) )        

            # Cross-channel masking
            if self.do_xchannel_masking:
                num_ch = M_pu.shape[0]
                M = torch.empty_like(M_pu)
                xcm_weights = torch.reshape( (2**self.xcm_weights), (4,4,1,1,1) )[:num_ch,...]
                for cc in range(num_ch): # for each channel: Sust, RG, VY, Trans
                    M[cc,...] = torch.sum( M_pu * xcm_weights[:,cc], dim=0, keepdim=True )
            else:
                M = M_pu

            D_u = self.mask_func_perc_norm( torch.abs(T-R), M )

            if self.masking_model == "soft_clamp_cont":
                D = D_u
            else:
                D = self.clamp_diffs( D_u )
        else:
            raise RuntimeError( f"Unknown masking model {self.masking_model}" )

        if self.debug and hasattr(self,"mem_allocated_peak"): 
            allocated = torch.cuda.memory_allocated(self.device)
            self.mem_allocated_peak = max( self.mem_allocated_peak, allocated )

        return D

    def clamp_diffs(self,D):
        if self.dclamp_type == "hard":
            Dc = torch.clamp(D, max=(10**self.d_max))
        elif self.dclamp_type == "soft":
            max_v = 10**self.d_max
            Dc = max_v * D / (max_v + D)
        elif self.dclamp_type == "none":
            Dc = D
        elif self.dclamp_type == "per_channel":
            num_ch = D.shape[0]
            max_v = 10**(self.d_max[:num_ch,...].view(-1,1,1,1))
            Dc = max_v * D / (max_v + D)
        else:
            raise RuntimeError( f"Unknown difference clamping type {self.dclamp_type}" )

        return Dc


    def phase_uncertainty(self, M):
        # Blur only when the image is larger then the required pad size
        if self.pu_dilate != 0 and M.shape[-2]>self.pu_padsize and M.shape[-1]>self.pu_padsize:
            #M_pu = utils.imgaussfilt( M, self.pu_dilate ) * torch.pow(10.0, self.mask_c)
            M_pu = self.pu_blur.forward(M) * (10**self.mask_c)
        else:
            M_pu = M * (10**self.mask_c)
        return M_pu

    def phase_uncertainty_no_c(self, M):
        # Blur only when the image is larger then the required pad size
        if self.pu_dilate != 0 and M.shape[-2]>self.pu_padsize and M.shape[-1]>self.pu_padsize:
            #M_pu = utils.imgaussfilt( M, self.pu_dilate ) * torch.pow(10.0, self.mask_c)
            M_pu = self.pu_blur.forward(M)
        else:
            M_pu = M
        return M_pu

    def mask_func_perc_norm(self, G, G_mask ):
        # Masking on perceptually normalized quantities (as in Daly's VDP)        
        p = self.mask_p
        if self.masking_model == "none":
            R = torch.pow(G,p)
        else:
            no_channels = G_mask.shape[0]
            if hasattr( self, 'mask_q' ):
                q = self.mask_q[0:no_channels].view(no_channels,1,1,1)
            else:
                q_sust = self.mask_q_sust.clamp(1.0, 7.0)
                q_trans = self.mask_q_trans.clamp(1.0, 7.0)
                if no_channels==3: # image
                    q = torch.stack( [q_sust, q_sust, q_sust], dim=0 ).view(3,1,1,1)
                else: # video
                    q = torch.stack( [q_sust, q_sust, q_sust, q_trans], dim=0 ).view(4,1,1,1)

            if self.masking_model == "smooth_clamp_cont":
                R = torch.div( self.smooth_clamp_cont(G, p), 1. + safe_pow(G_mask, q) )
            else:
                R = torch.div(safe_pow(G,p), 1. + safe_pow(G_mask, q))
        return R

    def smooth_clamp_cont( self, C, p ):
        max_v = 10**self.d_max
        C_clamped = torch.div( (max_v*(C**p)+1), (max_v + C**p) )
        return C_clamped


    def compute_local_contrast(self, T_f, R_f, lpyr, L_bkg_pyr, bb):
        if self.local_adapt=="simple_ref":
            L_bkg = lpyr.get_gband(L_bkg_pyr,bb)[1:2,:,:,:].clamp(min=0.01) # sustained, reference
            T = T_f / L_bkg  
            R = R_f / L_bkg
        else:
            raise RuntimeError( f"Error: local adaptation {self.local_adapt} not supported" )

        return L_bkg, T, R

    def weber2log(self, W):
        # Convert Weber contrast 
        #
        # W = (B-A)/A
        #
        # to log contrast
        #
        # G = log10( B/A );
        #
        return torch.log10(1.0 + W)

    def lp_norm(self, x, p, dim=0, normalize=True, keepdim=True):
        if dim is None:
            dim = 0

        if normalize:
            if isinstance(dim, tuple):
                N = 1.0
                for dd in dim:
                    N *= x.shape[dd]
            else:
                N = x.shape[dim]
        else:
            N = 1.0

        if isinstance( p, torch.Tensor ): 
            # p is a Tensor if it is being optimized. In that case, we need the formula for the norm
            return safe_pow( torch.sum( safe_pow(x, p), dim=dim, keepdim=keepdim)/float(N), 1/p) 
        else:
            return torch.norm(x, p, dim=dim, keepdim=keepdim) / (float(N) ** (1./p))

    # Return temporal filters
    # F[0] - Y sustained
    # F[1] - rg sustained
    # F[2] - yv sustained
    # F[3] - Y transient
    def get_temporal_filters(self, frames_per_s):

        N = int(math.ceil(0.250 * frames_per_s/2)*2)+1 # The length of the filter, always odd number
        N_omega = int(N/2)+1 # We need fewer freq coefficients as we use real FFT
        omega = torch.linspace( 0, frames_per_s/2, N_omega, device=self.device ).view(1,N_omega)

        R = torch.empty( (4, N_omega), device=self.device )
        # Sustained channels 
        R[0:3,:] = torch.exp( -omega ** self.beta_tf[0:3].view(3,1) / self.sigma_tf[0:3].view(3,1) )  # Freqency-space response
        # Transient channel

        omega_bands = torch.as_tensor( [0., 5.], device=self.device )
        if self.temp_filter == "hp_trans":
            # high-pass transient channel
            R[3:4,:] = 1-R[0:1,:]
        else:
            R[3:4,:] = torch.exp( -(omega ** self.beta_tf[3] - omega_bands[1] ** self.beta_tf[3])**2  / self.sigma_tf[3] )  # Freqency-space response

        #r = torch.empty( (4, N), device=self.device )

        F = []
        if self.device.type == 'mps':
            # FFT operations not supported on MPS as of torch==2.1 (see https://github.com/pytorch/pytorch/issues/78044)
            R = R.cpu()

        for kk in range(4):
            # Must be executed once per each channel. For some reason, gives wrong results when run on the entire array
            if self.temp_filter == "grad_trans" and kk==3:
                r = torch.zeros( (N), device=self.device )
                r[0] = 1
                r[2] = -1
            else:
                r = torch.fft.fftshift( torch.real( torch.fft.irfft( R[kk,:], norm="backward", n=N ) ) ).to(self.device)
            F.append( r )

        return F, omega_bands

    def short_name(self):
        return "cvvdp"

    def quality_unit(self):
        return "JOD"

    def get_info_string(self):
        if self.display_name.startswith('standard_'):
            #append this if are using one of the standard displays
            standard_str = self.display_name
        else:
            standard_str = f'custom-display: {self.display_name}'

        L_black, L_refl = self.display_photometry.get_black_level()
        return f'"ColourVideoVDP v{self.version}, {self.pix_per_deg:.4g} [pix/deg], ' \
               f'Lpeak={self.display_photometry.get_peak_luminance():.5g}, ' \
               f'Lblack={L_black:.4g}, Lrefl={L_refl:.4g} [cd/m^2], ({standard_str})"' 

    def write_features_to_json(self, stats, dest_fname):
        Q_per_ch = stats['Q_per_ch'] # quality per channel [cc,ff,bb]
        fmap = {}
        for key, value in stats.items():
            if not key in ["Q_per_ch", "heatmap"]:
                if isinstance(value, np.ndarray):
                    fmap[key] = value.tolist()
                else:
                    fmap[key] = value

        for cc in range(Q_per_ch.shape[0]): # for each temporal/chromatic channel
            for bb in range(Q_per_ch.shape[2]): # for each band
                fmap[f"t{cc}_b{bb}"] = Q_per_ch[cc,:,bb].tolist()

        with open(dest_fname, 'w', encoding='utf-8') as f:
            json.dump(fmap, f, ensure_ascii=False, indent=4)

    def save_to_config(self, fname, comment):
        # Save the current parameters to the given file
        assert fname.endswith('.json'), 'Please provide a .json file'
        parameters = utils.json2dict(self.parameters_file)
        for key in parameters:
            if isinstance(parameters[key], str):
                # strings remain the same
                continue
            elif isinstance(parameters[key], int):
                parameters[key] = getattr(self, key).item()
            elif isinstance(parameters[key], float):
                # np.float32 is not serializable
                parameters[key] = np.float64(getattr(self, key).item())
            elif isinstance(parameters[key], list):
                parameters[key] = list(getattr(self, 'ch_weights').detach().cpu().numpy().astype(np.float64))

        parameters['__comment'] = comment
        parameters['calibration_date'] = date.today().strftime('%d/%m/%Y')

        with open(fname, 'w') as f:
            json.dump(parameters, f, indent=4)


    # Export the visualization of distortions over time
    def export_distogram(self, stats, fname, jod_max=None, base_size=6):
        # Q_per_ch[channel,frame,sp_band]
        Q_per_ch = torch.as_tensor( stats['Q_per_ch'], device=self.device )
        ch_no = Q_per_ch.shape[0]    

        Q_per_ch[:,:,-1] *= self.baseband_weight[0:ch_no].view(-1,1,1)
        Q_per_ch *= self.get_ch_weights(ch_no)*ch_no
        dmap = (10. - self.met2jod(Q_per_ch)).cpu().numpy()

        if jod_max is None:
            jod_max = math.ceil(dmap.max())
        
        dmap /= jod_max

        fps = stats['frames_per_second']
        band_no = Q_per_ch.shape[2]
        frame_no = Q_per_ch.shape[1]
        rho_band = stats['rho_band']
        band_labels = [f"{val:.2f}" for val in np.flip(rho_band)[::2]]
        band_labels[0] = "BB"

        if not has_matplotlib:
            raise RuntimeError( 'matplotlib is missing. Please install it before exporting distograms.')
            
        fig, axs = plt.subplots(nrows=ch_no, figsize=(base_size*frame_no/60+0.5, base_size))

        ch_labels = ["A-sust", "RG", "YV", "A-trans"]
        cmap = plt.colormaps["plasma"]

        for kk in range(ch_no):
            dmap_ch = np.flip(np.transpose(dmap[kk,:,:].clip(0.,1.)),axis=0)
            axs[kk].imshow(dmap_ch, cmap=cmap, aspect="auto" )
            axs[kk].set_ylabel( ch_labels[kk] )
            axs[kk].yaxis.set_major_locator(ticker.FixedLocator(range(0,len(band_labels)*2,2)))
            axs[kk].yaxis.set_minor_locator(ticker.MultipleLocator(1.0))
            axs[kk].set_yticklabels(band_labels)
            if kk==(ch_no-1):
                axs[kk].xaxis.set_major_formatter(lambda x, pos: str(int(x/fps*1000)))
                axs[kk].set_xlabel( 'Time [ms]')
                axs[kk].xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
            else:
                axs[kk].set_xticks([])

        plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
        cax = plt.axes([0.925, 0.1, 0.025, 0.8])
        plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, jod_max), cmap=cmap), cax=cax, cmap=cmap)

        # fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
        #             ax=axs[0], label="JODs")

        plt.savefig( fname, bbox_inches='tight' )  

        # fig.show()
        # plt.waitforbuttonpress()        
        


