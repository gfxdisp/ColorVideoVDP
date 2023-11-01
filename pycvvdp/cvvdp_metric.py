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
        self.csf = parameters['csf']
        self.local_adapt = parameters['local_adapt'] # Local adaptation: 'simple' or or 'gpyr'
        self.contrast = parameters['contrast']  # One of: 'weber_g0_ref', 'weber_g1_ref', 'weber_g1', 'log'
        self.jod_a = torch.as_tensor( parameters['jod_a'], device=self.device )
        self.jod_exp = torch.as_tensor( parameters['jod_exp'], device=self.device )

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
        self.dclamp_type = parameters['dclamp_type']  # clamping mode: soft or hard
        self.dclamp_par = torch.as_tensor( parameters['dclamp_par'], device=self.device ) # Clamping of difference values
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
        per_sband_w = torch.ones( (1,1,no_bands), device=self.device)
        per_sband_w[0,0,-1] = self.baseband_weight

        #per_sband_w = torch.exp(interp1( self.quality_band_freq_log, self.quality_band_w_log, torch.log(torch.as_tensor(rho_band, device=self.device)) ))[:,None,None]

        Q_sc = self.lp_norm(Q_per_ch*per_ch_w*per_sband_w, self.beta_sch, dim=2, normalize=False)  # Sum across spatial channels

        is_image = (no_frames==1)
        t_int = self.image_int if is_image else 1.0 # Integration correction for images

        if not is_image and self.do_Bloch_int:
            bfilt_len = int(math.ceil(self.bfilt_duration * fps))
            Q_in = Q_sc.permute(0,2,1)
            B_filt = torch.ones( (1,1,bfilt_len), device=Q_in.device )/float(bfilt_len)
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
            Q = self.lp_norm(Q_tc,     self.beta_t,   dim=1, normalize=True)*t_int   # Sum across frames

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

            Q_per_ch_block[:,:,bb] = self.lp_norm(D, self.beta, dim=(-2,-1), normalize=True, keepdim=False) # Pool across all pixels (spatial pooling)

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

    def apply_masking_model(self, T, R, S):
        # T - test contrast tensor T[channel,frame,width,height]
        # R - reference contrast tensor
        # S - sensitivity
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

        D = self.clamp_diffs( self.mask_func_perc_norm( torch.abs(T-R), M ) )

        if self.debug and hasattr(self,"mem_allocated_peak"): 
            allocated = torch.cuda.memory_allocated(self.device)
            self.mem_allocated_peak = max( self.mem_allocated_peak, allocated )

        return D

    def clamp_diffs(self,D):
        if self.dclamp_type == "hard":
            Dc = torch.clamp(D, max=(10**self.dclamp_par))
        elif self.dclamp_type == "soft":
            max_v = 10**self.dclamp_par
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
            R = torch.div(torch.pow(G,p), 1. + torch.pow(G_mask, q))
        return R


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
            return torch.pow( torch.sum(x ** (p), dim=dim, keepdim=keepdim)/float(N), 1/p) 
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
        R[3:4,:] = torch.exp( -(omega ** self.beta_tf[3] - omega_bands[1] ** self.beta_tf[3])**2  / self.sigma_tf[3] )  # Freqency-space response

        #r = torch.empty( (4, N), device=self.device )

        F = []
        if self.device.type == 'mps':
            # FFT operations not supported on MPS as of torch==2.1 (see https://github.com/pytorch/pytorch/issues/78044)
            R = R.cpu()
        for kk in range(4):
            # Must be executed once per each channel. For some reason, gives wrong results when run on the entire array
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

        Q_per_ch[:,:,-1] *= self.baseband_weight
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
        

class cvvdp_image(vq_metric):
    def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, calibrated_ckpt=None):
        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        # Create a dummy display photometry object that does not change input frame
        self.cvvdp_metric = cvvdp(display_name, None, None, config_paths, heatmap, quiet, self.device, temp_padding, use_checkpoints, calibrated_ckpt)

    def set_display_model(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[]):
        self.linear_dm = vvdp_display_photo_eotf(display_photometry.Y_peak, contrast=display_photometry.contrast, source_colorspace='BT.2020-linear', E_ambient=display_photometry.E_ambient, k_refl=display_photometry.k_refl)
        self.cvvdp_metric.set_display_model(display_photometry=self.linear_dm, display_geometry=display_geometry)

    def predict_video_source(self, vid_source):

        _, _, N_frames = vid_source.get_video_size()

        avg = 0
        for ff in range(N_frames):
            T = vid_source.get_test_frame(ff, device=self.device, colorspace='RGB2020')
            R = vid_source.get_reference_frame(ff, device=self.device, colorspace='RGB2020')
            test_vs = video_source_array( T, R, fps=0, display_photometry=self.linear_dm )
            Q, _ = self.cvvdp_metric.predict_video_source(test_vs)
            avg += Q / N_frames

        return avg, None

    def short_name(self):
        return "cvvdp-image"

    def quality_unit(self):
        return "JOD"
