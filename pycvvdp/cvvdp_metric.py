from abc import abstractmethod
from urllib.parse import ParseResultBytes
from numpy.lib.shape_base import expand_dims
import torch
from torch.utils import checkpoint
from torch.functional import Tensor
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

from pycvvdp.visualize_diff_map import visualize_diff_map
from pycvvdp.video_source import *

# For debugging only
# from gfxdisp.pfs.pfs_torch import pfs_torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from third_party.cpuinfo import cpuinfo
from pycvvdp.fvvdp_lpyr_dec import fvvdp_lpyr_dec, fvvdp_contrast_pyr
from interp import interp1, interp3

import pycvvdp.utils as utils

#from utils import *
#from fvvdp_test import FovVideoVDP_Testbench

from pycvvdp.display_model import vvdp_display_photometry, vvdp_display_geometry
from pycvvdp.csf import castleCSF

"""
ColourVideoVDP metric. Refer to pytorch_examples for examples on how to use this class. 
"""
class cvvdp:
    def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, color_space="sRGB", foveated=False, heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False):
        self.quiet = quiet
        self.foveated = foveated
        self.heatmap = heatmap
        self.color_space = color_space
        self.temp_padding = temp_padding
        self.use_checkpoints = use_checkpoints # Used for training

        self.set_display_model(display_name, display_photometry=display_photometry, display_geometry=display_geometry)

        self.do_heatmap = (not self.heatmap is None) and (self.heatmap != "none")

        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        self.load_config()

        # if self.mask_s > 0.0:
        #     self.mask_p = self.mask_q + self.mask_s

        self.csf = castleCSF(device=self.device)

        # self.csf_cache              = {}
        # self.csf_cache_dirs         = [
        #                                 "csf_cache",
        #                                 os.path.join(os.path.dirname(__file__), "csf_cache"),
        #                               ]

        # self.omega = torch.tensor([0,5], device=self.device, requires_grad=False)
        # for oo in self.omega:
        #     self.preload_cache(oo, self.csf_sigma)

        self.lpyr = None
        self.imgaussfilt = utils.ImGaussFilt(0.5 * self.pix_per_deg, self.device)
        self.heatmap_pyr = None

    def update_device( self, device ):
        self.device = device
        self.omega = torch.tensor([0,5], device=self.device, requires_grad=False)
        for oo in self.omega:
            self.preload_cache(oo, self.csf_sigma)

        self.lpyr = None
        self.imgaussfilt = utils.ImGaussFilt(0.5 * self.pix_per_deg, self.device)
        # self.quality_band_freq_log = self.quality_band_freq_log.to(device)
        # self.quality_band_w_log = self.quality_band_w_log.to(device)

    def load_config( self ):

        #parameters_file = os.path.join(os.path.dirname(__file__), "fvvdp_data/fvvdp_parameters.json")
        self.parameters_file = utils.config_files.find( "cvvdp_parameters.json" )
        logging.debug( f"Loading ColourVideoVDP parameters from '{self.parameters_file}'" )
        parameters = utils.json2dict(self.parameters_file)

        #all common parameters between Matlab and Pytorch, loaded from the .json file
        self.mask_p = parameters['mask_p']
        self.mask_c = parameters['mask_c'] # content masking adjustment
        self.pu_dilate = parameters['pu_dilate']
        self.beta = parameters['beta'] # The exponent of the spatial summation (p-norm)
        self.beta_t = parameters['beta_t'] # The exponent of the summation over time (p-norm)
        self.beta_tch = parameters['beta_tch'] # The exponent of the summation over temporal channels (p-norm)
        self.beta_sch = parameters['beta_sch'] # The exponent of the summation over spatial channels (p-norm)
        self.sustained_sigma = parameters['sustained_sigma']
        self.sustained_beta = parameters['sustained_beta']
        self.csf_sigma = parameters['csf_sigma']
        self.sensitivity_correction = parameters['sensitivity_correction'] # Correct CSF values in dB. Negative values make the metric less sensitive.
        self.masking_model = parameters['masking_model']
        self.local_adapt = parameters['local_adapt'] # Local adaptation: 'simple' or or 'gpyr'
        self.contrast = parameters['contrast']  # Either 'weber' or 'log'
        self.jod_a = parameters['jod_a']
        self.jod_exp = parameters['jod_exp']
        self.mask_q_sust = parameters['mask_q_sust']
        self.mask_q_trans = parameters['mask_q_trans']
        self.filter_len = parameters['filter_len']
        self.ch_weights = torch.as_tensor( parameters['ch_weights'], device=self.device ) # Per-channel weight, Y-sust, rg, vy, Y-trans
        self.baseband_weight = parameters['baseband_weight']
        self.version = parameters['version']

        # other parameters
        self.debug = False

    def set_display_model(self, display_name="standard_4k", display_photometry=None, display_geometry=None):
        if display_photometry is None:
            self.display_photometry = vvdp_display_photometry.load(display_name)
            self.display_name = display_name
        else:
            self.display_photometry = display_photometry
            self.display_name = "unspecified"
        
        if display_geometry is None:
            self.display_geometry = vvdp_display_geometry.load(display_name)
        else:
            self.display_geometry = display_geometry

        self.pix_per_deg = self.display_geometry.get_ppd()

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

        test_vs = video_source_array( test_cont, reference_cont, frames_per_second, dim_order=dim_order, display_photometry=self.display_photometry, color_space_name=self.color_space )

        return self.predict_video_source(test_vs)

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays.
    '''
    def predict_video_source(self, vid_source):
        # We assume the pytorch default NCDHW layout

        vid_sz = vid_source.get_video_size() # H, W, F
        height, width, N_frames = vid_sz

        if self.lpyr is None or self.lpyr.W!=width or self.lpyr.H!=height:
            if self.local_adapt=="gpyr":
                self.lpyr = fvvdp_contrast_pyr(width, height, self.pix_per_deg, self.device)
            else:
                self.lpyr = fvvdp_lpyr_dec(width, height, self.pix_per_deg, self.device)

            if self.do_heatmap:
                self.heatmap_pyr = fvvdp_lpyr_dec(width, height, self.pix_per_deg, self.device)


        #assert self.W == R_vid.shape[-1] and self.H == R_vid.shape[-2]
        #assert len(R_vid.shape)==5

        is_image = (N_frames==1)  # Can run faster on images

        if is_image:
            temp_ch = 1  # How many temporal channels
            self.omega = [0]
        else:
            temp_ch = 2
            self.filter_len = int(np.ceil( 250.0 / (1000.0/vid_source.get_frames_per_second()) ))
            self.F, self.omega = self.get_temporal_filters(vid_source.get_frames_per_second())

        all_ch = 2+temp_ch

        if self.do_heatmap:
            dmap_channels = 1 if self.heatmap == "raw" else 3
            heatmap = torch.zeros([1,dmap_channels,N_frames,height,width], dtype=torch.float16, device=torch.device('cpu')) # Store heatmap in the CPU memory
        else:
            heatmap = None

        sw_buf = [None, None]
        Q_per_ch = None

        fl = self.filter_len

        if torch.cuda.is_available() and not is_image:
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
            mem_per_frame = pix_cnt*350   # Estimated memory required per frame

            max_frames = int((mem_avail-mem_const)/mem_per_frame) # how many frames can we fit into memory

            block_N_frames = max(1, min(max_frames,N_frames))  # Process so many frames in one pass 
            if self.debug: logging.debug( f"Processing {block_N_frames} frames in a batch." )
        else:
            block_N_frames = 1

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
                    sw_buf[0] = torch.empty((1,3,fl+block_N_frames-1,height,width), device=self.device, dtype=torch.float32) # TODO: switch to float16
                    sw_buf[1] = torch.empty((1,3,fl+block_N_frames-1,height,width), device=self.device, dtype=torch.float32)

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
                    sw_buf[0][:,:,0:-cur_block_N_frames,:,:] = sw_buf[0][:,:,cur_block_N_frames:,:,:]
                    sw_buf[1][:,:,0:-cur_block_N_frames,:,:] = sw_buf[1][:,:,cur_block_N_frames:,:,:]

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
                Q_per_ch_block = checkpoint.checkpoint(self.process_block_of_frames, R, vid_sz, temp_ch, heatmap, use_reentrant=False)
            else:
                Q_per_ch_block = self.process_block_of_frames(R, vid_sz, temp_ch, heatmap)

            if Q_per_ch is None:
                Q_per_ch = torch.zeros((Q_per_ch_block.shape[0], N_frames, Q_per_ch_block.shape[2]), device=self.device)
            
            Q_per_ch[:,ff:(ff+Q_per_ch_block.shape[1]),:] = Q_per_ch_block  

        rho_band = self.lpyr.get_freqs()
        Q_jod = self.do_pooling_and_jods(Q_per_ch, rho_band[0:-1])

        stats = {}
        stats['Q_per_ch'] = Q_per_ch.detach().cpu().numpy() # the quality per channel and per frame
        stats['rho_band'] = rho_band # Thespatial frequency per band
        stats['frames_per_second'] = vid_source.get_frames_per_second()
        stats['width'] = width
        stats['height'] = height
        stats['N_frames'] = N_frames

        if self.do_heatmap:            
            stats['heatmap'] = heatmap

        if self.debug and hasattr(self,"mem_allocated_peak"): 
            logging.debug( f"Allocated at start: {self.mem_allocated_start/1e9} GB" )
            logging.debug( f"Max allocated: {self.mem_allocated_peak/1e9} GB" )
            pix_cnt = width*height
            # sw_buf            
            mem_const = pix_cnt*4*3*2*(fl-1)
            per_pixel = (self.mem_allocated_peak-self.mem_allocated_start-mem_const)/(pix_cnt*block_N_frames)
            logging.debug( f"Memory used per pixel: {per_pixel} B" )

        return (Q_jod.squeeze(), stats)

    # Perform pooling with per-band weights and map to JODs
    def do_pooling_and_jods(self, Q_per_ch, rho_band):
        # Q_per_ch[channel,frame,sp_band]

        # Weights for the two temporal channels
        no_channels = Q_per_ch.shape[0]
        no_frames = Q_per_ch.shape[1]
        if no_frames>1: # If video
            per_ch_w = self.ch_weights[0:no_channels].view(-1,1,1)
            #torch.stack( (torch.ones(1, device=self.device), torch.as_tensor(self.w_transient, device=self.device)[None] ), dim=1)[:,:,None]
        else: # If image
            per_ch_w = 1

        # Weights for the spatial bands
        #per_sband_w = torch.exp(interp1( self.quality_band_freq_log, self.quality_band_w_log, torch.log(torch.as_tensor(rho_band, device=self.device)) ))[:,None,None]

        #Q_sc = self.lp_norm(Q_per_ch*per_tband_w*per_sband_w, self.beta_sch, 0, False)  # Sum across spatial channels
        Q_sc = self.lp_norm(Q_per_ch*per_ch_w, self.beta_sch, dim=2, normalize=False)  # Sum across spatial channels
        Q_tc = self.lp_norm(Q_sc,     self.beta_tch, dim=0, normalize=False)  # Sum across temporal and chromatic channels
        Q    = self.lp_norm(Q_tc,     self.beta_t,   dim=1, normalize=True)   # Sum across frames
        Q = Q.squeeze()
            
        Q_JOD = 10. - self.jod_a * Q**self.jod_exp
        return Q_JOD

        # sign = lambda x: (1, -1)[x<0]
        # beta_jod = 10.0**self.log_jod_exp
        # Q_jod = sign(self.jod_a) * ((abs(self.jod_a)**(1.0/beta_jod))* Q)**beta_jod + 10.0 # This one can help with very large numbers
        # return Q_jod.squeeze()

    def process_block_of_frames(self, R, vid_sz, temp_ch, heatmap):
        # R[channels,frames,width,height]
        #height, width, N_frames = vid_sz
        all_ch = 2+temp_ch

        # Perform Laplacian pyramid decomposition
        B_bands, L_bkg_pyr = self.lpyr.decompose(R[0,...])

        if self.debug: assert len(B_bands) == self.lpyr.get_band_count()

        # if self.do_heatmap:
        #     Dmap_pyr_bands, Dmap_pyr_gbands = self.heatmap_pyr.decompose( torch.zeros([1,1,height,width], dtype=torch.float, device=self.device))

        # L_bkg_bb = [None for i in range(self.lpyr.get_band_count()-1)]

        rho_band = self.lpyr.get_freqs()

        Q_per_ch_block = None
        block_N_frames = R.shape[-3] 

        for bb in range(self.lpyr.get_band_count()):  # For each spatial frequency band

            is_baseband = (bb==(self.lpyr.get_band_count()-1))

            T_f = self.lpyr.get_band(B_bands, bb)[0::2,...] # Test
            R_f = self.lpyr.get_band(B_bands, bb)[1::2,...] # Reference

            if is_baseband:
                L_bkg = torch.mean(R_f[0:1,...], dim=(-2,-1), keepdim=True)  # Use the mean from the reference sustained as the background luminance
                rho = 0.1
                S = torch.empty((all_ch,block_N_frames,1,1), device=self.device)
                for cc in range(all_ch):
                    S = self.csf.sensitivity(rho, self.omega[tch], L_bkg, cch, self.csf_sigma) * 10.0**(self.sensitivity_correction/20.0)

                D = ((T_f-R_f) / L_bkg * S) * self.baseband_weight
            else:
                if self.local_adapt=="gpyr":
                    L_bkg = self.lpyr.get_gband(L_bkg_pyr, bb) 
                else:
                    raise RuntimeError( "Not implemented")
                    # # 1:2 below is passing reference sustained
                    # L_bkg, R_f, T_f = self.compute_local_contrast(R_f, T_f, 
                    #     self.lpyr.get_gband(L_bkg_pyr, bb+1)[1:2,...], L_adapt)

                # Compute CSF
                rho = rho_band[bb] # Spatial frequency in cpd
                ch_height, ch_width = L_bkg.shape[-2], L_bkg.shape[-1]
                S = torch.empty((all_ch,block_N_frames,ch_height,ch_width), device=self.device)
                for cc in range(all_ch):
                    tch = 0 if cc<3 else 1  # Sustained or transient
                    cch = cc if cc<3 else 0 # Y, rg, yv
                    S[cc,:,:,:] = self.csf.sensitivity(rho, self.omega[tch], L_bkg, cch, self.csf_sigma) * 10.0**(self.sensitivity_correction/20.0)

                D = self.apply_masking_model(T_f, R_f, S)

            # if self.do_heatmap:
            #     if cc == 0: self.heatmap_pyr.set_band(Dmap_pyr_bands, bb, D)
            #     else:       self.heatmap_pyr.set_band(Dmap_pyr_bands, bb, self.heatmap_pyr.get_band(Dmap_pyr_bands, bb) + w_temp_ch[cc] * D)

            if Q_per_ch_block is None:
                Q_per_ch_block = torch.empty((all_ch, block_N_frames, self.lpyr.get_band_count()), device=self.device)

            Q_per_ch_block[:,:,bb] = self.lp_norm(D, self.beta, dim=(-2,-1), normalize=True, keepdim=False) # Pool across all pixels (spatial pooling)

        # if self.do_heatmap:
        #     beta_jod = np.power(10.0, self.log_jod_exp)
        #     dmap = torch.pow(self.heatmap_pyr.reconstruct(Dmap_pyr_bands), beta_jod) * abs(self.jod_a)         
        #     if self.heatmap == "raw":
        #         heatmap[:,:,ff,...] = dmap.detach().type(torch.float16).cpu()
        #     else:
        #         ref_frame = R[:,0, :, :, :]
        #         heatmap[:,:,ff,...] = visualize_diff_map(dmap, context_image=ref_frame, colormap_type=self.heatmap).detach().type(torch.float16).cpu()
        
        return Q_per_ch_block

    def apply_masking_model(self, T, R, S):
        T = T*S
        R = R*S
        M = self.phase_uncertainty( torch.min( torch.abs(T), torch.abs(R) ) )
        D = self.mask_func_perc_norm( torch.abs(T-R), M )
        D = torch.clamp(D, max=1e4)

        if self.debug and hasattr(self,"mem_allocated_peak"): 
            allocated = torch.cuda.memory_allocated(self.device)
            self.mem_allocated_peak = max( self.mem_allocated_peak, allocated )

        return D

    def phase_uncertainty(self, M):
        if self.pu_dilate != 0:
            M_pu = utils.imgaussfilt( M, self.pu_dilate ) * torch.pow(10.0, self.mask_c)
        else:
            M_pu = M * torch.pow(self.torch_scalar(10.0), self.torch_scalar(self.mask_c))
        return M_pu

    def mask_func_perc_norm(self, G, G_mask ):
        # Masking on perceptually normalized quantities (as in Daly's VDP)        
        p = self.mask_p
        if G_mask.shape[0]==3: # image
            q = torch.tensor( [self.mask_q_sust, self.mask_q_sust, self.mask_q_sust], device=self.device ).view(3,1,1,1)
        else: # video
            q = torch.tensor( [self.mask_q_sust, self.mask_q_sust, self.mask_q_sust, self.mask_q_trans], device=self.device ).view(4,1,1,1)
        R = torch.div(torch.pow(G,p), 1. + torch.pow(G_mask, q))
        return R


    def compute_local_contrast(self, R, T, next_gauss_band, L_adapt):
        if self.local_adapt=="simple":
            L_bkg = Func.interpolate(L_adapt.unsqueeze(0).unsqueeze(0), R.shape, mode='bicubic', align_corners=True)
            # L_bkg = torch.ones_like(R) * torch.mean(R)
            # np2img(l2rgb(L_adapt.unsqueeze(-1).cpu().data.numpy())/200.0).show()
            # np2img(l2rgb(L_bkg[0,0].unsqueeze(-1).cpu().data.numpy())/200.0).show()
        elif self.local_adapt=="gpyr":
            if self.contrast == "log":
                next_gauss_band = torch.pow(10.0, next_gauss_band)
            L_bkg = self.lpyr.gausspyr_expand(next_gauss_band, [R.shape[-2], R.shape[-1]])
        else:
            print("Error: local adaptation %s not supported" % self.local_adapt)
            return

        if self.contrast != "log":
            L_bkg_clamped = torch.clamp(L_bkg, min=0.1)
            T = torch.clamp(torch.div(T, L_bkg_clamped), max=1000.0)
            R = torch.clamp(torch.div(R, L_bkg_clamped), max=1000.0)

        return L_bkg, R, T

    # def get_cache_key(self, omega, sigma, k_cm):
    #     return ("o%g_s%g_cm%f" % (omega, sigma, k_cm)).replace('-', 'n').replace('.', '_')

    # def preload_cache(self, omega, sigma):
    #     key = self.get_cache_key(omega, sigma, self.k_cm)
    #     for csf_cache_dir in self.csf_cache_dirs:
    #         #fname = os.path.join(csf_cache_dir, key + '_cpu.mat')
    #         fname = os.path.join(csf_cache_dir, key + '_gpu0.mat')
    #         if os.path.isfile(fname):
    #             #lut = load_mat_dict(fname, "lut_cpu", self.device)
    #             lut = utils.load_mat_dict(fname, "lut", self.device)
    #             for k in lut:
    #                 lut[k] = torch.tensor(lut[k], device=self.device, requires_grad=False)
    #             self.csf_cache[key] = {"lut" : lut}
    #             break
    #     if key not in self.csf_cache:
    #         raise RuntimeError("Error: cache file for %s not found" % key)

    # def cached_sensitivity(self, rho, omega, L_bkg, ecc, sigma):
    #     key = self.get_cache_key(omega, sigma, self.k_cm)

    #     if key in self.csf_cache:
    #         lut = self.csf_cache[key]["lut"]
    #     else:
    #         print("Error: Key %s not found in cache" % key)

    #     # ASSUMPTION: rho_q and ecc_q are not scalars
    #     rho_q = torch.log2(torch.clamp(rho,   lut["rho"][0], lut["rho"][-1]))
    #     Y_q   = torch.log2(torch.clamp(L_bkg, lut["Y"][0],   lut["Y"][-1]))
    #     ecc_q = torch.sqrt(torch.clamp(ecc,   lut["ecc"][0], lut["ecc"][-1]))

    #     interpolated = interp3( lut["rho_log"], lut["Y_log"], lut["ecc_sqrt"], lut["S_log"], rho_q, Y_q, ecc_q)

    #     S = torch.pow(2.0, interpolated)

    #     return S

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

        return torch.norm(x, p, dim=dim, keepdim=keepdim) / (float(N) ** (1./p))

    # Return temporal filters
    # F[0] - Y sustained
    # F[1] - rg sustained
    # F[2] - yv sustained
    # F[3] - Y transient
    def get_temporal_filters(self, frames_per_s):
        t = torch.linspace(0.0, self.filter_len / frames_per_s, self.filter_len, device=self.device)
        F = torch.zeros((4, t.shape[0]), device=self.device)

        sigma = torch.tensor([self.sustained_sigma], device=self.device)
        beta = torch.tensor([self.sustained_beta], device=self.device)

        F[0] = torch.exp(-torch.pow(torch.log(t + 1e-4) - torch.log(beta), 2.0) / (2.0 * (sigma ** 2.0)))
        F[0] = F[0] / torch.sum(F[0])

        Fdiff = F[0, 1:] - F[0, :-1]

        # TODO: Implement sustained colour channels
        F[1] = F[0]
        F[2] = F[0]

        k2 = 0.062170507756932
        # This one seems to be slightly more accurate at low sampling rates
        F[3] = k2*torch.cat([Fdiff/(t[1]-t[0]), torch.tensor([0.0], device=self.device)], 0) # transient

        omega = torch.tensor([0,5], device=self.device, requires_grad=False)

        F[0].requires_grad = False
        F[3].requires_grad = False

        return F, omega

    def torch_scalar(self, val, dtype=torch.float32):
        return torch.tensor(val, dtype=dtype, device=self.device)

    def short_name(self):
        return "FovVideoVDP"

    def quality_unit(self):
        return "JOD"

    def get_info_string(self):
        if self.display_name.startswith('standard_'):
            #append this if are using one of the standard displays
            standard_str = ', (' + self.display_name + ')'
        else:
            standard_str = ''
        fv_mode = 'foveated' if self.foveated else 'non-foveated'
        return '"FovVideoVDP v{}, {:.4g} [pix/deg], Lpeak={:.5g}, Lblack={:.4g} [cd/m^2], {}{}"'.format(self.version, self.pix_per_deg, self.display_photometry.get_peak_luminance(), self.display_photometry.get_black_level(), fv_mode, standard_str)

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


