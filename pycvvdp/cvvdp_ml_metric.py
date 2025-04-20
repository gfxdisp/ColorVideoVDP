from abc import abstractmethod
from urllib.parse import ParseResultBytes
try:
    from numpy import expand_dims
except ImportError:
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
import torch.utils.benchmark as torchbench
import logging
from datetime import date
from torchvision.ops import MLP

try:
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from matplotlib.colors import Normalize
    has_matplotlib = True
except:
    has_matplotlib = False

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    has_nvml = True
except:
    has_nvml = False

from pycvvdp.visualize_diff_map import visualize_diff_map
from pycvvdp.video_source import *

from pycvvdp.vq_metric import *

from pycvvdp.dump_channels import DumpChannels

from pycvvdp.cvvdp_metric import cvvdp

#from pycvvdp.colorspace import lms2006_to_dkld65

# For debugging only
# from gfxdisp.pfs.pfs_torch import pfs_torch

from pycvvdp.lpyr_dec import lpyr_dec, lpyr_dec_2, weber_contrast_pyr, log_contrast_pyr
from interp import interp1, interp3, interp1dim2

import pycvvdp.utils as utils

from pycvvdp.display_model import vvdp_display_photometry, vvdp_display_geometry
from pycvvdp.csf import castleCSF


class cvvdp_feature_pooling(torch.nn.Module):

    def __init__(self, feature_size):
        super().__init__()

        self.avg_pool = torch.nn.AvgPool2d( (feature_size,feature_size), ceil_mode=True )

    def forward(self, T, R, D):
        # T - test
        # R - reference
        # D - difference
        # T[channels,frames,width,height]
        # F[frames,width,height,channels,stat]
        
        dim_order = [1, 2, 3, 0] # put channels as the last dimension
        mean_T = self.avg_pool( T ).permute(dim_order)
        var_T = self.avg_pool( T**2 ).permute(dim_order) - mean_T**2
        mean_R = self.avg_pool( R ).permute(dim_order)
        var_R = self.avg_pool( R**2 ).permute(dim_order) - mean_R**2
        mean_D = self.avg_pool( D ).permute(dim_order)
        var_D = self.avg_pool( D**2 ).permute(dim_order) - mean_D**2

        F = torch.stack( (mean_T, var_T, mean_R, var_R, mean_D, var_D), dim=4 )

        assert(not F.isnan().any())

        return F


"""
ColorVideoVDP metric with ML head.
"""
class cvvdp_ml(cvvdp):

    # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
    # random_init - do not load NN from a checkpoint file, use a random initialization
    def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.random_init = random_init

        self.disabled_features = disabled_features        

        dropout = 0.2
        hidden_dims = 24
        num_layers = 3
        ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
        stats_no = 2 # 6 extracted stats - for now do 2
        self.feature_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(device)

        super().__init__(display_name=display_name, display_photometry=display_photometry,
                         display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
                         quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
                         dump_channels=dump_channels, gpu_mem=gpu_mem)


    # Switch to training mode (e.g., to optimize memory allocation)
    def train(self, do_training=True):
        super().train(do_training)
        self.feature_net.train(do_training)

    # So that we can override in the super classes
    def get_nets_to_load(self):
        return [ 'feature_net' ]

    def load_config( self, config_paths ):
        super().load_config(config_paths)

        if not self.random_init:
            # Load the checkpoint for NN
            ckpt_file = utils.config_files.find( "cvvdp.ckpt", config_paths )

            logging.info( f"Loading cvvdp checkpoint file from {ckpt_file}" )

            for net in self.get_nets_to_load():
                prefix = net + '.'
                if torch.cuda.is_available():
                    state_dict = {key[len(prefix):]: val for key, val in torch.load(ckpt_file, map_location=self.device)['state_dict'].items() if key.startswith(prefix)}
                else:
                    state_dict = {key[len(prefix):]: val for key, val in torch.load(ckpt_file, map_location=torch.device('cpu'))['state_dict'].items() if key.startswith(prefix)}
                getattr(self, net).load_state_dict(state_dict)
                #.to(device=self.device)

    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays. Video source is recommended when processing long videos as it allows frame-by-frame loading.
    '''
    def predict_video_source(self, vid_source):
        # We assume the pytorch default NCDHW layout

        features, heatmap = self.extract_features(vid_source)

        Q_jod = self.do_pooling_and_jods(features)

        vid_sz = vid_source.get_video_size() # H, W, F
        height, width, N_frames = vid_sz

        stats = {}
        rho_band = self.lpyr.get_freqs()
        stats['rho_band'] = rho_band # The spatial frequency per band in cpd        
        fps = vid_source.get_frames_per_second()
        stats['frames_per_second'] = fps
        stats['width'] = width
        stats['height'] = height
        stats['N_frames'] = N_frames

        if self.dump_channels:
            self.dump_channels.close()

        if self.do_heatmap:            
            stats['heatmap'] = heatmap

        return (Q_jod.squeeze(), stats)


    def extract_features(self, vid_source):

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
            pix_cnt = width*height
            block_N_frames = self.estimate_block_N(pix_cnt, N_frames)
        else:
            block_N_frames = 1

        if self.contrast=="log":
            met_colorspace='logLMS_DKLd65'
        else:
            met_colorspace='DKLd65' # This metric uses DKL colourspaxce with d65 whitepoint

        if self.dump_channels:
            self.dump_channels.open(vid_source.get_frames_per_second())

        # Spatial size of a feature patch in 1 visual degree
        feature_size = math.floor(self.pix_per_deg) 

        features = None

        for ff in range(0, N_frames, block_N_frames):
            cur_block_N_frames = min(block_N_frames,N_frames-ff) # How many frames in this block?

            if is_image:                
                R = torch.empty((1, 6, 1, height, width), device=self.device)
                R[:,0::2, :, :, :] = vid_source.get_test_frame(0, device=self.device, colorspace=met_colorspace)
                R[:,1::2, :, :, :] = vid_source.get_reference_frame(0, device=self.device, colorspace=met_colorspace)

            else: # This is video
                #if self.debug: print("Frame %d:\n----" % ff)

                if ff == 0: # First frame
                    sw_buf[0] = torch.zeros((1,3,fl+block_N_frames-1,height,width), device=self.device, dtype=torch.float16) # TODO: switch to float16
                    sw_buf[1] = torch.zeros((1,3,fl+block_N_frames-1,height,width), device=self.device, dtype=torch.float16)
                    #print( f"Allocated {sw_buf[0].nelement()*sw_buf[0].element_size()/1e9*2} GB for {fl+block_N_frames-1} frame buffer.")

                    if self.debug and not hasattr( self, 'sw_buf_allocated' ):
                        # Memory allocated after creating buffers for temporal filters 
                        self.sw_buf_allocated = torch.cuda.max_memory_allocated(self.device)

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

            if self.dump_channels:
                self.dump_channels.dump_temp_ch(R)

            if self.use_checkpoints:
                # Used for training
                features_per_block, heatmap_block = checkpoint.checkpoint(self.process_block_of_frames, R, temp_ch, self.lpyr, is_image, use_reentrant=False)
            else:
                features_per_block, heatmap_block = self.process_block_of_frames(R, temp_ch, self.lpyr, is_image)

            
            if features is None:
                features = [None] * len(features_per_block)
                for bb in range(len(features_per_block)):
                    features[bb] = torch.empty((N_frames, features_per_block[bb].shape[1], features_per_block[bb].shape[2], features_per_block[bb].shape[3], features_per_block[bb].shape[4]), device=self.device)

            ff_end = ff+features_per_block[bb].shape[0]
            for bb in range(len(features_per_block)):
                features[bb][ff:ff_end,:,:,:,:] = features_per_block[bb]

            if self.do_heatmap:
                if self.heatmap == "raw":
                    heatmap[:,:,ff:ff_end,...] = heatmap_block.detach().type(torch.float16).cpu()
                else:
                    ref_frame = R[:,0, :, :, :]
                    heatmap[:,:,ff:ff_end,...] = visualize_diff_map(heatmap_block, context_image=ref_frame, colormap_type=self.heatmap, use_cpu=self.device.type == 'mps').detach().type(torch.float16).cpu()

        return features, heatmap


    # Perform pooling with per-band weights and map to JODs
    def do_pooling_and_jods(self, features):

        # features[band][frames,width,height,channels,stat]
        # disables_features is an array of indices of the stat to be disabled

        # no_channels = features[0].shape[3]
        # no_frames = features[0].shape[0]
        no_bands = len(features)

        Q_JOD = torch.as_tensor(10., device=self.device)

        is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

        for bb in range(no_bands):

            #F[frames,width,height,channels,stat]
            f = features[bb]
            # Remove unecessary features (for now) - keep only mean D and std D
            f = f[:, :, :, :, 4:]
            f[:, :, :, :, 1] = torch.sqrt(torch.abs(f[:, :, :, :, 1]))

            if is_image:
                f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
            if self.disabled_features is not None:
                f[:, :, :, :, self.disabled_features] = 0  
                # f[:, :, :, :, 5] = torch.sqrt(torch.abs(f[:, :, :, :, 5]))  
            f = f.flatten( start_dim=3 )
            D_all = self.feature_net(f)

            is_base_band = (bb==no_bands-1)
            if is_base_band:
                D_all *= self.baseband_weight

            if is_image:
                D_all *= self.image_int

            Q_JOD -= D_all.view(-1).mean()/no_bands

        assert(not Q_JOD.isnan())
        return Q_JOD

    def process_block_of_frames(self, R, temp_ch, lpyr, is_image):
        # R[channels,frames,width,height]
        all_ch = 2+temp_ch

        #torch.autograd.set_detect_anomaly(True)

        # Perform Laplacian pyramid decomposition
        B_bands, L_bkg_pyr = lpyr.decompose(R[0,...])

        if self.debug: assert len(B_bands) == lpyr.get_band_count()

        if self.dump_channels:
            self.dump_channels.dump_lpyr(lpyr, B_bands)

        rho_band = lpyr.get_freqs()
        rho_band[lpyr.get_band_count()-1] = 0.1 # Baseband

        features_block = None
        block_N_frames = R.shape[-3] 
        N_bands = lpyr.get_band_count()

        features_block = [None] * N_bands        

        for bb in range(N_bands):  # For each spatial frequency band

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

            #width = R.shape[-1]
            #feature_size = math.ceil(self.pix_per_deg * ch_width / width)
            feature_size = math.ceil(self.pix_per_deg)

            fp = cvvdp_feature_pooling(feature_size)
            features_block[bb] = fp( torch.abs(T_f)*S, torch.abs(R_f)*S, D )

            # if bb>6:
            #     Q_per_ch_block[:,:,bb] = 0

            if self.do_heatmap:

                # We need to reduce the differences across the channels using the right weights
                # Weights for the channels: sustained, RG, YV, [transient]
                t_int = self.image_int if is_image else 1.0
                per_ch_w = self.get_ch_weights( all_ch ).view(-1,1,1,1) * t_int
                if is_baseband:
                    per_ch_w *= self.baseband_weight[0:all_ch].view(-1,1,1,1)

                D_chr = self.lp_norm(D*per_ch_w, self.beta_tch, dim=-4, normalize=False)  # Sum across temporal and chromatic channels
                self.heatmap_pyr.set_lband(bb, D_chr)

            if self.dump_channels:
                width = R.shape[-1]
                height = R.shape[-2]
                t_int = self.image_int if is_image else 1.0
                per_ch_w = self.get_ch_weights( all_ch ).view(-1,1,1,1) * t_int
                self.dump_channels.set_diff_band(width, height, lpyr.ppd, bb, D*per_ch_w)

        if self.do_heatmap:
            heatmap_block = 1.-(self.met2jod( self.heatmap_pyr.reconstruct() )/10.)
        else:
            heatmap_block = None

        if self.dump_channels:
            self.dump_channels.dump_diff()

        return features_block, heatmap_block


# Adds an attention module to the cvvdp_ml
class cvvdp_ml_att(cvvdp_ml):

    # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
    # random_init - do not load NN from a checkpoint file, use a random initialization
    def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

        dropout = 0.2
        hidden_dims = 48
        num_layers = 4
        ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
        stats_no = 4 # T, T_var, R, R_var
        self.att_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(device)

        super().__init__(display_name=display_name, display_photometry=display_photometry,
                         display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
                         quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
                         dump_channels=dump_channels, gpu_mem=gpu_mem, random_init=random_init, disabled_features=disabled_features)

    def get_nets_to_load(self):
        return [ 'feature_net', 'att_net' ]

    # Perform pooling with per-band weights and map to JODs
    def do_pooling_and_jods(self, features):

        # features[band][frames,width,height,channels,stat]
        # disables_features is an array of indices of the stat to be disabled

        # no_channels = features[0].shape[3]
        # no_frames = features[0].shape[0]
        no_bands = len(features)

        Q_JOD = torch.as_tensor(10., device=self.device)

        is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

        for bb in range(no_bands):

            #F[frames,width,height,channels,stat]
            f = features[bb]
            
            # Variance into std
            f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

            if is_image:
                f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
            if self.disabled_features is not None:
                f[:, :, :, :, self.disabled_features] = 0  

            f_TR = f[:, :, :, :, 0:4].flatten( start_dim=3 )
            f_D = f[:, :, :, :, 4:].flatten( start_dim=3 )

            Att = self.att_net(f_TR)
            D_all = self.feature_net(f_D) * Att

            is_base_band = (bb==no_bands-1)
            if is_base_band:
                D_all *= self.baseband_weight

            if is_image:
                D_all *= self.image_int

            Q_JOD -= D_all.view(-1).mean()/no_bands

        assert(not Q_JOD.isnan())
        return Q_JOD
