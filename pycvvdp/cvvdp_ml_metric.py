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
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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

from pycvvdp.cvvdp_metric import cvvdp, safe_pow

#from pycvvdp.colorspace import lms2006_to_dkld65

# For debugging only
# from gfxdisp.pfs.pfs_torch import pfs_torch

from pycvvdp.lpyr_dec import lpyr_dec, lpyr_dec_2, weber_contrast_pyr, log_contrast_pyr
from interp import interp1, interp3, interp1dim2

import pycvvdp.utils as utils

from pycvvdp.display_model import vvdp_display_photometry, vvdp_display_geometry
from pycvvdp.csf import castleCSF

from huggingface_hub import hf_hub_download
os.environ["HF_HUB_TOKEN"] = ""  # empty string disables token


class cvvdp_avg_pool(torch.nn.AvgPool2d):

    def forward( self, X ):
        V = X.view((-1,)+X.shape[2:])  # Need to combine batch and channel so that AvgPool2d works
        Y = super().forward(V)
        return Y.view( X.shape[0:2] + Y.shape[1:] )



class cvvdp_feature_pooling(torch.nn.Module):

    def __init__(self, feature_size):
        super().__init__()

        self.avg_pool = cvvdp_avg_pool( (feature_size,feature_size), ceil_mode=True )

    def forward(self, T, R, D):
        # T - test
        # R - reference
        # D - difference
        # T[batch,channels,frames,width,height]
        # F[batch,frames,width,height,channels,stat]
                
        dim_order = [0, 2, 3, 4, 1] # put channels as the last dimension
        mean_T = self.avg_pool( T ).permute(dim_order)
        var_T = self.avg_pool( T**2 ).permute(dim_order) - mean_T**2
        mean_R = self.avg_pool( R ).permute(dim_order)
        var_R = self.avg_pool( R**2 ).permute(dim_order) - mean_R**2
        mean_D = self.avg_pool( D ).permute(dim_order)
        var_D = self.avg_pool( D**2 ).permute(dim_order) - mean_D**2

        F = torch.stack( (mean_T, var_T, mean_R, var_R, mean_D, var_D), dim=5 )

        assert(not F.isnan().any())

        return F


"""
Base class for all ColorVideoVDP with ML heads
"""
class cvvdp_ml_base(cvvdp):

    # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
    # random_init - do not load NN from a checkpoint file, use a random initialization
    def __init__(self, random_init = False, disabled_features=None, **kwargs):

        self.random_init = random_init
        self.disabled_features = disabled_features        

        super().__init__(**kwargs)

    def set_device( self, device ):
        if hasattr( self, "device" ):
            return

        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device


    # Switch to training mode (e.g., to optimize memory allocation)
    def train(self, do_training=True):
        super().train(do_training)
        for net in self.get_nets_to_load():
            getattr(self, net).train(do_training)

    # So that we can override in the super classes
    @abstractmethod
    def get_nets_to_load(self):
        """
        """

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

    def predict_video_source(self, vid_source):
        # We assume the pytorch default NCDHW layout

        assert vid_source.get_batch_size()==1 or self.heatmap is None or self.heatmap=='none', 'Heatmaps not supported when batches are used'

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
        batch_sz = vid_source.get_batch_size()

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
                R = torch.empty((batch_sz, 6, 1, height, width), device=self.device)
                R[:,0::2, :, :, :] = vid_source.get_test_frame(0, device=self.device, colorspace=met_colorspace)
                R[:,1::2, :, :, :] = vid_source.get_reference_frame(0, device=self.device, colorspace=met_colorspace)

            else: # This is video
                #if self.debug: print("Frame %d:\n----" % ff)

                if ff == 0: # First frame
                    sw_buf[0] = torch.zeros((batch_sz,3,fl+block_N_frames-1,height,width), device=self.device, dtype=torch.float16) # TODO: switch to float16
                    sw_buf[1] = torch.zeros((batch_sz,3,fl+block_N_frames-1,height,width), device=self.device, dtype=torch.float16)
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
                R = torch.zeros((batch_sz, 8, cur_block_N_frames, height, width), device=self.device)

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
                    features[bb] = torch.empty(( (batch_sz,N_frames) + features_per_block[bb].shape[2:]), device=self.device)

            ff_end = ff+features_per_block[bb].shape[0]
            for bb in range(len(features_per_block)):
                features[bb][:,ff:ff_end,:,:,:,:] = features_per_block[bb]

            if self.do_heatmap:
                if self.heatmap == "raw":
                    heatmap[:,:,ff:ff_end,...] = heatmap_block.detach().type(torch.float16).cpu()
                else:
                    ref_frame = R[:,0, :, :, :]
                    heatmap[:,:,ff:ff_end,...] = visualize_diff_map(heatmap_block, context_image=ref_frame, colormap_type=self.heatmap, use_cpu=self.device.type == 'mps').detach().type(torch.float16).cpu()

        return features, heatmap


    # Perform pooling with per-band weights and map to JODs
    @abstractmethod
    def do_pooling_and_jods(self, features):
        """
        """

    def process_block_of_frames(self, R, temp_ch, lpyr, is_image):
        # R[batch,channels,frames,width,height]
        all_ch = 2+temp_ch
        batch_sz = R.shape[0]

        #torch.autograd.set_detect_anomaly(True)

        # Perform Laplacian pyramid decomposition
        B_bands, L_bkg_pyr = lpyr.decompose(R)

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
            T_f = B_bb[:,0::2,...] # Test
            R_f = B_bb[:,1::2,...] # Reference

            logL_bkg = lpyr.get_gband(L_bkg_pyr, bb)

            # Compute CSF
            rho = rho_band[bb] # Spatial frequency in cpd
            ch_height, ch_width = logL_bkg.shape[-2], logL_bkg.shape[-1]
            S = torch.empty((batch_sz,all_ch,block_N_frames,ch_height,ch_width), device=self.device)
            for cc in range(all_ch):
                tch = 0 if cc<3 else 1  # Sustained or transient
                cch = cc if cc<3 else 0 # Y, rg, yv
                # The sensitivity is always extracted for the reference frame
                S[:,cc:(cc+1),:,:,:] = self.csf.sensitivity(rho, self.omega[tch], logL_bkg[...,1:2,:,:,:], cch, self.csf_sigma) * 10.0**(self.sensitivity_correction/20.0)

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


"""
ColorVideoVDP metric with an MLP head.
"""
class cvvdp_ml(cvvdp_ml_base):

    # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
    # random_init - do not load NN from a checkpoint file, use a random initialization
    def __init__(self, device=None, **kwargs):

        self.set_device( device )

        dropout = 0.2
        hidden_dims = 24
        num_layers = 3
        ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
        stats_no = 2 # 6 extracted stats - for now do 2
        self.feature_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

        super().__init__(device=device, **kwargs)

    # So that we can override in the super classes
    def get_nets_to_load(self):
        return [ 'feature_net' ]

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

            if is_image:
                f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
            if self.disabled_features is not None:
                f[:, :, :, :, self.disabled_features] = 0  

            # We want to only keep mean D and std D in this version
            f = f[:, :, :, :, 4:]
            f[:, :, :, :, 1] = torch.sqrt(torch.abs(f[:, :, :, :, 1]))

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

    def export_distogram(self, stats, fname, jod_max=None, base_size=6):
        raise NotImplementedError        


# Adds a saliency module to the cvvdp_ml
class cvvdp_ml_saliency(cvvdp_ml):

    # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
    # random_init - do not load NN from a checkpoint file, use a random initialization
    def __init__(self, config_paths=[], device=None, **kwargs):

        self.set_device( device )

        dropout = 0.2
        hidden_dims = 48
        num_layers = 4
        ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
        stats_no = 4 # T, T_var, R, R_var
        self.att_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

        path = os.path.join(os.path.dirname(__file__), "vvdp_data", "cvvdp_ml_saliency")
        met_config_paths = config_paths.copy() # We do not want to modify config_path for other metrics
        met_config_paths.append( path )

        # Downloads the file if not cached; returns local path to cached file
        model_path = hf_hub_download(
            repo_id="gfxdisp/cvvdp_ml",
            filename="cvvdp_ml_saliency/cvvdp.ckpt"
        )
        met_config_paths.append(os.path.dirname(model_path))


        super().__init__(config_paths=met_config_paths, device=device, **kwargs)

    def get_nets_to_load(self):
        return [ 'feature_net', 'att_net' ]

    # Perform pooling with per-band weights and map to JODs
    def do_pooling_and_jods(self, features):

        # features[band][batch,frames,width,height,channels,stat]
        # disables_features is an array of indices of the stat to be disabled

        # no_channels = features[0].shape[3]
        # no_frames = features[0].shape[0]
        no_bands = len(features)
        batch_sz = features[0].shape[0]

        Q_JOD = torch.ones((batch_sz), device=self.device)*10.

        is_image = (features[0].shape[4]==3) # if 3 channels, it is an image

        for bb in range(no_bands):

            #F[batch,frames,width,height,channels,stat]
            f = features[bb]
            
            # Variance into std
            f[...,1::2] = torch.sqrt(torch.abs(f[...,1::2]))

            if is_image:
                f = torch.cat( (f, torch.zeros((f.shape[0:4] + (1,f.shape[5])), device=self.device)), dim=4) # Add the missing channel
            if self.disabled_features is not None:
                f[..., self.disabled_features] = 0  

            f_TR = f[..., 0:4].flatten( start_dim=4 )
            f_D = f[..., 4:].flatten( start_dim=4 )

            Att = self.att_net(f_TR)
            Att = F.relu(Att)
            D_all = self.feature_net(f_D) 
            D_all = F.relu(D_all) * Att /no_bands

            is_base_band = (bb==no_bands-1)
            if is_base_band:
                D_all *= self.baseband_weight

            if is_image:
                D_all *= self.image_int

            Q_JOD -= self.spatiotemporal_pooling(D_all)

        assert(not Q_JOD.isnan().any())
        return Q_JOD

    def full_name(self):
        return "ColorVideoVDP-ML-Saliency"

    def spatiotemporal_pooling(self, D_all):
        return D_all.view(D_all.shape[0],-1).mean(dim=1)
    

register_metric( cvvdp_ml_saliency )


class RegressionTransformer(nn.Module):
    def __init__(self,
                 in_channels=32,  # TR(16) + D(8)
                 patch_size=16,
                 dim=256,
                 depth=4,
                 heads=8,
                 dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.patch_embed = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.Linear(in_channels, dim),
            #nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b h w c -> b (h w) c')
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim*4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=depth
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.ReLU()
        )
    def forward(self, x):
        # x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.patch_embed(x)  # [B, N_patches, dim]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        cls_feat = x[:, 0]
        return self.reg_head(cls_feat).squeeze(-1)
    
    def get_heatmap(self, x):
        # x: [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.patch_embed(x)  # [B, N_patches, dim]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        cls_feat = x[:, 1:]
        return self.reg_head(cls_feat).squeeze(-1)


class cvvdp_ml_transformer(cvvdp_ml):
    def __init__(self,
                 patch_size=(9, 16),
                 dim=256,
                 config_paths=[],
                 **kwargs):
        
        self.set_device( kwargs.get('device') )
        
        met_config_paths = config_paths.copy() # We do not want to modify config_path for other metrics
        path = os.path.join(os.path.dirname(__file__), "vvdp_data", "cvvdp_ml_transformer")
        met_config_paths.append( path )

        # Downloads the file if not cached; returns local path to cached file
        model_path = hf_hub_download(
            repo_id="gfxdisp/cvvdp_ml",
            filename="cvvdp_ml_transformer/cvvdp.ckpt"
        )
        met_config_paths.append(os.path.dirname(model_path))

        self.transformer_net = RegressionTransformer(
            in_channels=24,  # TR(4*4) + D(2*4)
            patch_size=patch_size,
            dim=dim
        ).to(self.device)

        super().__init__(config_paths=met_config_paths, **kwargs)

    def get_nets_to_load(self):
        return ['transformer_net']
    
    def do_pooling_and_jods(self, features):

        Q_JOD = torch.as_tensor(10., device=self.device)
        is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

        for bb, f in enumerate(features):

            f[..., 1::2] = torch.sqrt(torch.abs(f[..., 1::2]))

            if is_image:
                f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
            if self.disabled_features is not None:
                f[..., self.disabled_features] = 0

            f_all = torch.cat([
                f[..., 0:4].flatten(start_dim=3),
                f[..., 4:].flatten(start_dim=3)
            ], dim=-1)

            delta = self.transformer_net(f_all) / len(features)

            if bb == len(features)-1:
                delta *= self.baseband_weight
            if is_image:
                delta *= self.image_int

            Q_JOD -= delta.mean()

        return Q_JOD

    def full_name(self):
        return "ColorVideoVDP-ML-Transformer"


register_metric( cvvdp_ml_transformer )


# """
# ColorVideoVDP metric with ML head as a no-reference metric.
# """
# class cvvdp_ml_nr(cvvdp_ml_base):

#     # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
#     # random_init - do not load NN from a checkpoint file, use a random initialization
#     def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

#         self.set_device( device )

#         dropout = 0.2
#         hidden_dims = 48
#         num_layers = 6
#         ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
#         stats_no = 2 # 6 extracted stats 
#         self.feature_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

#         super().__init__(display_name=display_name, display_photometry=display_photometry,
#                          display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
#                          quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
#                          dump_channels=dump_channels, gpu_mem=gpu_mem,
#                          random_init=random_init, disabled_features=disabled_features)

#     # So that we can override in the super classes
#     def get_nets_to_load(self):
#         return [ 'feature_net' ]

#     # Perform pooling with per-band weights and map to JODs
#     def do_pooling_and_jods(self, features):
#         # features[band][frames,width,height,channels,stat]
#         # disables_features is an array of indices of the stat to be disabled

#         # no_channels = features[0].shape[3]
#         # no_frames = features[0].shape[0]
#         no_bands = len(features)

#         Q_JOD = torch.as_tensor(0., device=self.device)

#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         for bb in range(no_bands):

#             #F[frames,width,height,channels,stat]
#             f = features[bb]

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
#             if self.disabled_features is not None:
#                 f[:, :, :, :, self.disabled_features] = 0  

#             f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

#             f = f[:, :, :, :, :2].flatten( start_dim=3 )

#             D_all = self.feature_net(f)

#             is_base_band = (bb==no_bands-1)
#             if is_base_band:
#                 D_all *= self.baseband_weight

#             if is_image:
#                 D_all *= self.image_int

#             Q_JOD += D_all.view(-1).mean()/no_bands

#         assert(not Q_JOD.isnan())
#         return Q_JOD

# register_metric( cvvdp_ml_nr )


# """
# ColorVideoVDP metric with an MLP head.
# """
# class cvvdp_ml_trd(cvvdp_ml_base):

#     # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
#     # random_init - do not load NN from a checkpoint file, use a random initialization
#     def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

#         self.set_device( device )

#         dropout = 0.2
#         hidden_dims = 48
#         num_layers = 6
#         ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
#         stats_no = 6 # 6 extracted stats 
#         self.feature_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

#         super().__init__(display_name=display_name, display_photometry=display_photometry,
#                          display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
#                          quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
#                          dump_channels=dump_channels, gpu_mem=gpu_mem,
#                          random_init=random_init, disabled_features=disabled_features)

#     # So that we can override in the super classes
#     def get_nets_to_load(self):
#         return [ 'feature_net' ]

#     # Perform pooling with per-band weights and map to JODs
#     def do_pooling_and_jods(self, features):
#         # features[band][frames,width,height,channels,stat]
#         # disables_features is an array of indices of the stat to be disabled

#         # no_channels = features[0].shape[3]
#         # no_frames = features[0].shape[0]
#         no_bands = len(features)

#         Q_JOD = torch.as_tensor(10., device=self.device)

#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         for bb in range(no_bands):

#             #F[frames,width,height,channels,stat]
#             f = features[bb]

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
#             if self.disabled_features is not None:
#                 f[:, :, :, :, self.disabled_features] = 0  

#             f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

#             f = f.flatten( start_dim=3 )
#             D_all = self.feature_net(f)

#             is_base_band = (bb==no_bands-1)
#             if is_base_band:
#                 D_all *= self.baseband_weight

#             if is_image:
#                 D_all *= self.image_int

#             Q_JOD -= D_all.view(-1).mean()/no_bands

#         assert(not Q_JOD.isnan())
#         return Q_JOD

# register_metric( cvvdp_ml_trd )


# """
# Use information from T and R to get texture similarity
# """
# class cvvdp_ml_texture_sim(cvvdp_ml_base):

#     # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
#     # random_init - do not load NN from a checkpoint file, use a random initialization
#     def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

#         self.set_device( device )

#         dropout = 0.2
#         hidden_dims = 24
#         num_layers = 6
#         ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
#         stats_no = 2 # 6 extracted stats 
#         self.feature_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)


#         super().__init__(display_name=display_name, display_photometry=display_photometry,
#                          display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
#                          quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
#                          dump_channels=dump_channels, gpu_mem=gpu_mem,
#                          random_init=random_init, disabled_features=disabled_features)
        
#     # So that we can override in the super classes
#     def get_nets_to_load(self):
#         return [ 'feature_net' ]

#     # Perform pooling with per-band weights and map to JODs
#     def do_pooling_and_jods(self, features):
#         # features[band][frames,width,height,channels,stat]
#         # disables_features is an array of indices of the stat to be disabled

#         # no_channels = features[0].shape[3]
#         # no_frames = features[0].shape[0]
#         no_bands = len(features)

#         Q_JOD = torch.as_tensor(10., device=self.device)

#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         for bb in range(no_bands):

#             #F[frames,width,height,channels,stat]
#             f = features[bb]
            
#             # Variance into std
#             f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
#             if self.disabled_features is not None:
#                 f[:, :, :, :, self.disabled_features] = 0  

#             # Get similarity of means and stds between T and R
#             distance = 0.5 * (f[:, :, :, :, 0] - f[:, :, :, :, 2])**2 + 0.5 * (f[:, :, :, :, 1] - f[:, :, :, :, 3])**2

#             # Follow what we did before
#             f = torch.sqrt( f[:, :, :, :, 4:] * distance.unsqueeze(-1) )
#             # f[:, :, :, :, 0] = torch.sqrt( f[:, :, :, :, 0] * distance )
#             # f[:, :, :, :, 1] = torch.sqrt( f[:, :, :, :, 1] * distance )

#             f = f.flatten( start_dim=3 )
#             D_all = self.feature_net(f)

#             is_base_band = (bb==no_bands-1)
#             if is_base_band:
#                 D_all *= self.baseband_weight

#             if is_image:
#                 D_all *= self.image_int

#             Q_JOD -= D_all.view(-1).mean()/no_bands

#         assert(not Q_JOD.isnan())
#         return Q_JOD

# register_metric( cvvdp_ml_texture_sim )


# """
# Use Distance between T and R to inform the prediction
# """
# class cvvdp_ml_dis_TR(cvvdp_ml_base):

#     # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
#     # random_init - do not load NN from a checkpoint file, use a random initialization
#     def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

#         self.set_device( device )

#         dropout = 0.2
#         hidden_dims = 24
#         num_layers = 6
#         ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
#         stats_no = 4 # mean D, std D, distance mean, distance std
#         self.feature_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

#         super().__init__(display_name=display_name, display_photometry=display_photometry,
#                          display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
#                          quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
#                          dump_channels=dump_channels, gpu_mem=gpu_mem,
#                          random_init=random_init, disabled_features=disabled_features)
    
#     # So that we can override in the super classes
#     def get_nets_to_load(self):
#         return [ 'feature_net' ]

#     # Perform pooling with per-band weights and map to JODs
#     def do_pooling_and_jods(self, features):
#         # features[band][frames,width,height,channels,stat]
#         # disables_features is an array of indices of the stat to be disabled

#         # no_channels = features[0].shape[3]
#         # no_frames = features[0].shape[0]
#         no_bands = len(features)

#         Q_JOD = torch.as_tensor(10., device=self.device)

#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         for bb in range(no_bands):

#             #F[frames,width,height,channels,stat]
#             f = features[bb]
            
#             # Variance into std
#             f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
#             if self.disabled_features is not None:
#                 f[:, :, :, :, self.disabled_features] = 0  

#             # Get similarity of means and stds between T and R as other features
#             f[:, :, :, :, 2] = torch.sqrt( (f[:, :, :, :, 0] - f[:, :, :, :, 2])**2 )
#             f[:, :, :, :, 3] = torch.sqrt( (f[:, :, :, :, 1] - f[:, :, :, :, 3])**2 )

#             # Remove first 2 stats, as they are no longer interesting
#             f = f[:, :, :, :, 2:].flatten( start_dim=3 )
#             D_all = self.feature_net(f)

#             is_base_band = (bb==no_bands-1)
#             if is_base_band:
#                 D_all *= self.baseband_weight

#             if is_image:
#                 D_all *= self.image_int

#             Q_JOD -= F.relu(D_all).view(-1).mean()/no_bands

#         assert(not Q_JOD.isnan())
#         return Q_JOD
    
# register_metric( cvvdp_ml_dis_TR )

# """
# Use Distance between T and R - normalized to inform the prediction
# """
# class cvvdp_ml_dis_TR_normalised(cvvdp_ml_base):

#     # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
#     # random_init - do not load NN from a checkpoint file, use a random initialization
#     def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

#         self.set_device( device )

#         dropout = 0.2
#         hidden_dims = 24
#         num_layers = 6
#         ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
#         stats_no = 4 # mean D, std D, distance mean, distance std
#         self.feature_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

#         super().__init__(display_name=display_name, display_photometry=display_photometry,
#                          display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
#                          quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
#                          dump_channels=dump_channels, gpu_mem=gpu_mem,
#                          random_init=random_init, disabled_features=disabled_features)
    
#     # So that we can override in the super classes
#     def get_nets_to_load(self):
#         return [ 'feature_net' ]

#     # Perform pooling with per-band weights and map to JODs
#     def do_pooling_and_jods(self, features):
#         # features[band][frames,width,height,channels,stat]
#         # disables_features is an array of indices of the stat to be disabled

#         # no_channels = features[0].shape[3]
#         # no_frames = features[0].shape[0]
#         no_bands = len(features)

#         Q_JOD = torch.as_tensor(10., device=self.device)

#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         for bb in range(no_bands):

#             #F[frames,width,height,channels,stat]
#             f = features[bb]
            
#             # Variance into std
#             f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
#             if self.disabled_features is not None:
#                 f[:, :, :, :, self.disabled_features] = 0  
            
#             c = 1e-6

#             # Get similarity of means and stds between T and R as other features
#             f[:, :, :, :, 2] = torch.sqrt( ( (f[:, :, :, :, 0] - f[:, :, :, :, 2])**2 + c ) / (f[:, :, :, :, 0]**2 + f[:, :, :, :, 2]**2 + c ) )
#             f[:, :, :, :, 3] = torch.sqrt( ( (f[:, :, :, :, 1] - f[:, :, :, :, 3])**2 + c ) / (f[:, :, :, :, 1]**2 + f[:, :, :, :, 3]**2 + c ) )

#             # Remove first 2 stats, as they are no longer interesting
#             f = f[:, :, :, :, 2:].flatten( start_dim=3 )
#             D_all = self.feature_net(f)

#             is_base_band = (bb==no_bands-1)
#             if is_base_band:
#                 D_all *= self.baseband_weight

#             if is_image:
#                 D_all *= self.image_int

#             Q_JOD -= D_all.view(-1).mean()/no_bands

#         assert(not Q_JOD.isnan())
#         return Q_JOD
    
# register_metric( cvvdp_ml_dis_TR_normalised )

# """
# Use Similarity between T and R to inform the prediction
# """
# class cvvdp_ml_sim_TR(cvvdp_ml_base):

#     # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
#     # random_init - do not load NN from a checkpoint file, use a random initialization
#     def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

#         self.set_device( device )

#         dropout = 0.2
#         hidden_dims = 24
#         num_layers = 6
#         ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
#         stats_no = 4 # mean D, std D, distance mean, distance std
#         self.feature_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

#         super().__init__(display_name=display_name, display_photometry=display_photometry,
#                          display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
#                          quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
#                          dump_channels=dump_channels, gpu_mem=gpu_mem,
#                          random_init=random_init, disabled_features=disabled_features)
    
#     # So that we can override in the super classes
#     def get_nets_to_load(self):
#         return [ 'feature_net' ]

#     # Perform pooling with per-band weights and map to JODs
#     def do_pooling_and_jods(self, features):
#         # features[band][frames,width,height,channels,stat]
#         # disables_features is an array of indices of the stat to be disabled

#         # no_channels = features[0].shape[3]
#         # no_frames = features[0].shape[0]
#         no_bands = len(features)

#         Q_JOD = torch.as_tensor(10., device=self.device)

#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         for bb in range(no_bands):

#             #F[frames,width,height,channels,stat]
#             f = features[bb]
            
#             # Variance into std
#             f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
#             if self.disabled_features is not None:
#                 f[:, :, :, :, self.disabled_features] = 0  

#             # Get similarity of means and stds between T and R as other features
#             mean_T = f[:, :, :, :, 0]
#             mean_R = f[:, :, :, :, 2]
#             std_T = f[:, :, :, :, 1]
#             std_R = f[:, :, :, :, 3]

#             c1 = 1e-6
#             f[:, :, :, :, 2] = 1 - ( (2*mean_T*mean_R + c1) / ((mean_T**2) + (mean_R)**2 + c1) )
#             f[:, :, :, :, 3] = 1 - ( (2*std_T*std_R + c1) / ((std_T**2) + (std_R)**2 + c1) )

#             # Remove first 2 stats, as they are no longer interesting
#             f = f[:, :, :, :, 2:].flatten( start_dim=3 )
#             D_all = self.feature_net(f)

#             is_base_band = (bb==no_bands-1)
#             if is_base_band:
#                 D_all *= self.baseband_weight

#             if is_image:
#                 D_all *= self.image_int

#             Q_JOD -= D_all.view(-1).mean()/no_bands

#         assert(not Q_JOD.isnan())
#         return Q_JOD

# register_metric( cvvdp_ml_sim_TR )



# # Mimics cvvdp pooling of differences but also weights the final predictions by learned saliency
# class cvvdp_ml_dpool_sal(cvvdp_ml_base):

#     # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
#     # random_init - do not load NN from a checkpoint file, use a random initialization
#     def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

#         self.set_device( device )

#         dropout = 0.2
#         hidden_dims = 12
#         num_layers = 3
#         ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
#         stats_no = 4 # T, T_var, R, R_var
#         self.att_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

#         super().__init__(display_name=display_name, display_photometry=display_photometry,
#                          display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
#                          quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
#                          dump_channels=dump_channels, gpu_mem=gpu_mem, random_init=random_init, disabled_features=disabled_features)

#     def get_nets_to_load(self):
#         return ['att_net'] #[ 'feature_net', 'att_net' ]

#     # Perform pooling with per-band weights and map to JODs
#     def do_pooling_and_jods(self, features):

#         # features[band][frames,width,height,channels,stat]
#         # disables_features is an array of indices of the stat to be disabled

#         # no_channels = features[0].shape[3]
#         # no_frames = features[0].shape[0]
#         no_bands = len(features)

#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         D_b = torch.as_tensor(0., device=self.device)
#         for bb in range(no_bands):

#             #F[frames,width,height,channels,stat]
#             f = features[bb]
            
#             # Variance into std
#             #f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
#             if self.disabled_features is not None:
#                 f[:, :, :, :, self.disabled_features] = 0  

#             f_TR = f[:, :, :, :, 0:4].flatten( start_dim=3 )

#             Att = self.att_net(f_TR)
#             Att = F.relu(Att)
#             epsilon = 1e-8
#             Att = Att / (torch.sum( Att, dim=(1,2), keepdim=True ) + epsilon) # Normalize in the spatial dimension
#             #Att = F.sigmoid(Att)
#             assert not Att.isnan().any() and Att.isfinite().all(), "NaNs or Infs in Att"

#             D_sp = self.lp_norm(f[:, :, :, :, 4]*Att, self.beta, dim=(1, 2), normalize=False, keepdim=True)  # Sum across all patches

#             per_ch_w = self.get_ch_weights( 4 ).view(1,1,1,-1)

#             D_ch = self.lp_norm(per_ch_w*D_sp, self.beta_tch, dim=3, normalize=False, keepdim=True)  # Sum across achromatic and chromatic channels

#             if is_image:
#                 D_t = D_ch * self.image_int
#             else:
#                 D_t = self.lp_norm(D_ch, self.beta_t, dim=0, normalize=True)   # Sum across frames

#             D_b += safe_pow(D_t.squeeze(), self.beta_sch)

#         D = safe_pow(D_b, 1/self.beta_sch)

#         Q_JOD = self.met2jod(D)            

#         assert(not Q_JOD.isnan())
#         return Q_JOD

# register_metric( cvvdp_ml_dpool_sal )


# # Polynomial regression
# class cvvdp_ml_poly_reg(cvvdp_ml_base):

#     # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
#     # random_init - do not load NN from a checkpoint file, use a random initialization
#     def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

#         self.set_device( device )

#         N_v = 24
#         N = N_v*2 + round(N_v*(N_v-1)/2)
#         self.poly_k = torch.randn( (1,1,1,N,1), device=device )

#         super().__init__(display_name=display_name, display_photometry=display_photometry,
#                          display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
#                          quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
#                          dump_channels=dump_channels, gpu_mem=gpu_mem, random_init=random_init, disabled_features=disabled_features)

#     def get_nets_to_load(self):
#         return [] 

#     # Perform pooling with per-band weights and map to JODs
#     def do_pooling_and_jods(self, features):

#         # features[band][frames,width,height,channels,stat]
#         # disables_features is an array of indices of the stat to be disabled

#         # no_channels = features[0].shape[3]
#         # no_frames = features[0].shape[0]
#         no_bands = len(features)

#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         D = torch.as_tensor(10., device=self.device)
#         for bb in range(no_bands):

#             #F[frames,width,height,channels,stat]
#             f = features[bb]
            
#             # Variance into std
#             f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel

#             if self.disabled_features is not None:
#                 f[:, :, :, :, self.disabled_features] = 0  

#             f_TR = f.flatten( start_dim=3 )

#             # Polynomial basis (2nd order only)

#             # Mixed products
#             N_v = f_TR.shape[3]
#             N = round(N_v*(N_v-1)/2)
#             f_TR_mixed = torch.empty( (f_TR.shape[0], f_TR.shape[1], f_TR.shape[2], N), device=self.device)
#             pp = 0
#             for rr in range(N_v):
#                 for cc in range(rr+1,N_v):
#                     f_TR_mixed[:,:,:,pp] = f_TR[:,:,:,rr] * f_TR[:,:,:,cc]
#                     pp += 1

#             f_poly = torch.cat( (f_TR, f_TR_mixed, f_TR**2), dim=3 )

#             D -= F.relu( self.spatiotemporal_pooling( torch.matmul( f_poly[:,:,:,None,:], self.poly_k ) ) )

#         assert(not D.isnan())
#         return D

#     def spatiotemporal_pooling(self, D_all):
#         return D_all.view(-1).mean()

# register_metric( cvvdp_ml_poly_reg )

# # Adds an attention module to the cvvdp_ml
# class cvvdp_ml_att_sim_TR(cvvdp_ml_dis_TR):

#     # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
#     # random_init - do not load NN from a checkpoint file, use a random initialization
#     def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

#         self.set_device( device )

#         dropout = 0.2
#         hidden_dims = 48
#         num_layers = 4
#         ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
#         stats_no = 4 # T, T_var, R, R_var
#         self.att_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

#         super().__init__(display_name=display_name, display_photometry=display_photometry,
#                          display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
#                          quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
#                          dump_channels=dump_channels, gpu_mem=gpu_mem, random_init=random_init, disabled_features=disabled_features)

#     def get_nets_to_load(self):
#         return [ 'feature_net', 'att_net' ]

#     # Perform pooling with per-band weights and map to JODs
#     def do_pooling_and_jods(self, features):

#         # features[band][frames,width,height,channels,stat]
#         # disables_features is an array of indices of the stat to be disabled

#         # no_channels = features[0].shape[3]
#         # no_frames = features[0].shape[0]
#         no_bands = len(features)

#         Q_JOD = torch.as_tensor(10., device=self.device)

#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         for bb in range(no_bands):

#             #F[frames,width,height,channels,stat]
#             f = features[bb]
            
#             # Variance into std
#             f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
#             if self.disabled_features is not None:
#                 f[:, :, :, :, self.disabled_features] = 0  

#             f_TR = f[:, :, :, :, 0:4].flatten( start_dim=3 )

#             f[:, :, :, :, 2] = torch.sqrt( (f[:, :, :, :, 0] - f[:, :, :, :, 2])**2 )
#             f[:, :, :, :, 3] = torch.sqrt( (f[:, :, :, :, 1] - f[:, :, :, :, 3])**2 )

#             # Remove first 2 stats, as they are no longer interesting
#             f_D = f[:, :, :, :, 2:].flatten( start_dim=3 )

#             Att = self.att_net(f_TR)
#             D_all = self.feature_net(f_D) * Att /no_bands

#             is_base_band = (bb==no_bands-1)
#             if is_base_band:
#                 D_all *= self.baseband_weight

#             if is_image:
#                 D_all *= self.image_int

#             Q_JOD -= self.spatiotemporal_pooling(D_all)

#         assert(not Q_JOD.isnan())
#         return Q_JOD

#     def spatiotemporal_pooling(self, D_all):
#         return D_all.view(-1).mean()

# register_metric( cvvdp_ml_att_sim_TR )



# # Adds an attention module to the cvvdp_ml
# class cvvdp_ml_att_sim_TR_v2(cvvdp_ml_sim_TR):

#     # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
#     # random_init - do not load NN from a checkpoint file, use a random initialization
#     def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

#         self.set_device( device )

#         dropout = 0.2
#         hidden_dims = 48
#         num_layers = 4
#         ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
#         stats_no = 4 # T, T_var, R, R_var
#         self.att_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

#         super().__init__(display_name=display_name, display_photometry=display_photometry,
#                          display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
#                          quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
#                          dump_channels=dump_channels, gpu_mem=gpu_mem, random_init=random_init, disabled_features=disabled_features)

#     def get_nets_to_load(self):
#         return [ 'feature_net', 'att_net' ]

#     # Perform pooling with per-band weights and map to JODs
#     def do_pooling_and_jods(self, features):

#         # features[band][frames,width,height,channels,stat]
#         # disables_features is an array of indices of the stat to be disabled

#         # no_channels = features[0].shape[3]
#         # no_frames = features[0].shape[0]
#         no_bands = len(features)

#         Q_JOD = torch.as_tensor(10., device=self.device)

#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         for bb in range(no_bands):

#             #F[frames,width,height,channels,stat]
#             f = features[bb]
            
#             # Variance into std
#             f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
#             if self.disabled_features is not None:
#                 f[:, :, :, :, self.disabled_features] = 0  

#             f_TR = f[:, :, :, :, 0:4].flatten( start_dim=3 )

#             mean_T = f[:, :, :, :, 0]
#             mean_R = f[:, :, :, :, 2]
#             std_T = f[:, :, :, :, 1]
#             std_R = f[:, :, :, :, 3]

#             c1 = 1e-6
#             f[:, :, :, :, 2] = 1 - ( (2*mean_T*mean_R + c1) / ((mean_T**2) + (mean_R)**2 + c1) )
#             f[:, :, :, :, 3] = 1 - ( (2*std_T*std_R + c1) / ((std_T**2) + (std_R)**2 + c1) )

#             # Remove first 2 stats, as they are no longer interesting
#             f_D = f[:, :, :, :, 2:].flatten( start_dim=3 )

#             Att = self.att_net(f_TR)
#             D_all = self.feature_net(f_D) * Att /no_bands

#             is_base_band = (bb==no_bands-1)
#             if is_base_band:
#                 D_all *= self.baseband_weight

#             if is_image:
#                 D_all *= self.image_int

#             Q_JOD -= self.spatiotemporal_pooling(D_all)

#         assert(not Q_JOD.isnan())
#         return Q_JOD

#     def spatiotemporal_pooling(self, D_all):
#         return D_all.view(-1).mean()

# register_metric( cvvdp_ml_att_sim_TR_v2 )


# # Adds a masking module to the cvvdp_ml
# class cvvdp_ml_masking_sim(cvvdp_ml_trd):

#     # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
#     # random_init - do not load NN from a checkpoint file, use a random initialization
#     def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

#         self.set_device( device )

#         dropout = 0.2
#         hidden_dims = 24
#         num_layers = 3
#         ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
#         stats_no = 2 # T, T_var, R, R_var
#         self.masking_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

#         super().__init__(display_name=display_name, display_photometry=display_photometry,
#                          display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
#                          quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
#                          dump_channels=dump_channels, gpu_mem=gpu_mem, random_init=random_init, disabled_features=disabled_features)

#     def get_nets_to_load(self):
#         return [ 'feature_net', 'masking_net' ]

#     # Perform pooling with per-band weights and map to JODs
#     def do_pooling_and_jods(self, features):

#         # features[band][frames,width,height,channels,stat]
#         # disables_features is an array of indices of the stat to be disabled

#         # no_channels = features[0].shape[3]
#         # no_frames = features[0].shape[0]
#         no_bands = len(features)

#         Q_JOD = torch.as_tensor(10., device=self.device)

#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         for bb in range(no_bands):

#             #F[frames,width,height,channels,stat]
#             f = features[bb]
            
#             # Variance into std
#             f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
#             if self.disabled_features is not None:
#                 f[:, :, :, :, self.disabled_features] = 0  
            
#             f_d = f[:, :, :, :, 4:].flatten( start_dim=3 )

#             mean_sim = (f[:, :, :, :, 0] - f[:, :, :, :, 2])**2
#             std_sim = (f[:, :, :, :, 1] - f[:, :, :, :, 3])**2
#             f_sim = torch.stack((mean_sim, std_sim), axis=-1).flatten( start_dim=3 )

#             mask = self.masking_net(f_sim)
#             D_all = self.feature_net(f_d) * mask /no_bands

#             is_base_band = (bb==no_bands-1)
#             if is_base_band:
#                 D_all *= self.baseband_weight

#             if is_image:
#                 D_all *= self.image_int

#             Q_JOD -= self.spatiotemporal_pooling(D_all)

#         assert(not Q_JOD.isnan())
#         return Q_JOD

#     def spatiotemporal_pooling(self, D_all):
#         return D_all.view(-1).mean()

# register_metric( cvvdp_ml_masking_sim )

# # Adds a recurrent network to pool visual differences over time
# class cvvdp_ml_recur_lstm(cvvdp_ml_base):

#     # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
#     # random_init - do not load NN from a checkpoint file, use a random initialization
#     def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], heatmap=None, quiet=False, device=None, temp_padding="replicate", use_checkpoints=False, dump_channels=None, gpu_mem = None, random_init = False, disabled_features=None):

#         self.set_device( device )

#         dropout = 0.1
#         input_dims_pooling = 8 # 2 stats * 4 channels
#         hidden_dims = 16
#         num_layers = 1
#         proj_size = 8
#         self.pooling_net = torch.nn.LSTM(input_dims_pooling, hidden_dims, num_layers, dropout=dropout, batch_first=False, proj_size=proj_size).to(device)                

#         dropout = 0.2
#         hidden_dims = 24
#         num_layers = 3
#         ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
#         stats_no = 2 # 6 extracted stats - for now do 2
#         self.feature_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

#         super().__init__(display_name=display_name, display_photometry=display_photometry,
#                          display_geometry=display_geometry, config_paths=config_paths, heatmap=heatmap,
#                          quiet=quiet, device=device, temp_padding=temp_padding, use_checkpoints=use_checkpoints,
#                          dump_channels=dump_channels, gpu_mem=gpu_mem, random_init=random_init, disabled_features=disabled_features)


#     def get_nets_to_load(self):
#         return [ 'pooling_net', 'feature_net' ]
    
#     # Perform pooling with per-band weights and map to JODs
#     def do_pooling_and_jods(self, features):

#         # features[band][frames,width,height,channels,stat]
#         # disables_features is an array of indices of the stat to be disabled

#         # no_channels = features[0].shape[3]
#         # no_frames = features[0].shape[0]
#         no_bands = len(features)

#         Q_JOD = torch.as_tensor(10., device=self.device)

#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         for bb in range(no_bands):

#             #F[frames,width,height,channels,stat]
#             f = features[bb]
            
#             # Variance into std
#             f[:, :, :, :, 1::2] = torch.sqrt(torch.abs(f[:, :, :, :, 1::2]))

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
#             if self.disabled_features is not None:
#                 f[:, :, :, :, self.disabled_features] = 0  

#             f_D = f[:, :, :, :, 4:].flatten( start_dim=3 )

#             # f_D[frames,width,height,8]
#             D = f_D.view( f_D.shape[0], -1, 8 )
#             D_temp, _ = self.pooling_net(D)  # LSTM to convert features into quality scores, the sequence is over time
#             D_mlp = self.feature_net(D_temp)  # We want only positive predictions
#             D_all = D_mlp.view(-1).mean()  # Spatial and temporal pooling

#             is_base_band = (bb==no_bands-1)
#             if is_base_band:
#                 D_all *= self.baseband_weight

#             if is_image:
#                 D_all *= self.image_int

#             Q_JOD -= D_all

#             assert(not Q_JOD.isnan())
#             return Q_JOD


    


# class RegressionTransformerPositionalEmbedding(nn.Module):
#     def __init__(self,
#                  in_channels=32,  # TR(16) + D(8)
#                  dim=256,
#                  depth=4,
#                  heads=8,
#                  dropout=0.1):
#         super().__init__()
#         self.dim = dim

#         self.patch_embed = nn.Sequential(
#             nn.Linear(in_channels, dim),
#             Rearrange('b h w c -> b (h w) c')
#         )

#         self.pos_embed_mlp = nn.Sequential(
#             nn.Linear(2, dim//2),
#             nn.GELU(),
#             nn.Linear(dim//2, dim)
#         )

#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

#         self.transformer = nn.TransformerEncoder(
#             encoder_layer=nn.TransformerEncoderLayer(
#                 d_model=dim,
#                 nhead=heads,
#                 dim_feedforward=dim*4,
#                 dropout=dropout,
#                 activation='gelu',
#                 batch_first=True,
#                 norm_first=True
#             ),
#             num_layers=depth
#         )

#         self.reg_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, 1),
#             nn.ReLU()
#         )

#     def get_position_embedding(self, h, w, device):
#         y_coords = (torch.arange(h, device=device).float() + 0.5) / h
#         x_coords = (torch.arange(w, device=device).float() + 0.5) / w
#         grid = torch.stack(torch.meshgrid(x_coords, y_coords, indexing='xy'), dim=-1)  # [H, W, 2]
        
#         pos_embed = self.pos_embed_mlp(grid)  # [H, W, dim]
#         return pos_embed.view(1, h*w, self.dim)  # [1, N_patches, dim]

#     def forward(self, x):
#         # x: [B, H, W, C]
#         B, H, W, C = x.shape
#         x = self.patch_embed(x)  # [B, N_patches, dim]
            
#         pos_embed = self.get_position_embedding(H, W, x.device)
#         x += pos_embed
        
#         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = self.transformer(x)
#         cls_feat = x[:, 0]
#         return self.reg_head(cls_feat).squeeze(-1)

    
# class cvvdp_ml_transformer_positional_embedding(cvvdp_ml_base):
#     def __init__(self,
#                  dim=256,
#                  **kwargs):
        
#         self.set_device( kwargs.get('device') )
        
#         self.transformer_net = RegressionTransformerPositionalEmbedding(
#             in_channels=24,  # TR(4*4) + D(2*4)
#             dim=dim
#         ).to(self.device)

#         super().__init__(**kwargs)

#     def get_nets_to_load(self):
#         return ['transformer_net']
    
#     def do_pooling_and_jods(self, features):

#         Q_JOD = torch.as_tensor(10., device=self.device)
#         is_image = (features[0].shape[3]==3) # if 3 channels, it is an image

#         for bb, f in enumerate(features):

#             f[..., 1::2] = torch.sqrt(torch.abs(f[..., 1::2]))

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3) # Add the missing channel
#             if self.disabled_features is not None:
#                 f[..., self.disabled_features] = 0

#             f = f.flatten( start_dim=3 )

#             delta = self.transformer_net(f) / len(features)

#             if bb == len(features)-1:
#                 delta *= self.baseband_weight
#             if is_image:
#                 delta *= self.image_int

#             Q_JOD -= delta.mean()

#         return Q_JOD

# register_metric( cvvdp_ml_transformer_positional_embedding )



# class RegressionTransformer_bands(nn.Module):
#     def __init__(self,
#                  in_channels=24,
#                  dim=256,
#                  depth=4,
#                  heads=8,
#                  dropout=0.1):
        
#         super().__init__()
#         self.dim = dim
        
#         self.patch_embed = nn.Sequential(
#             #Rearrange('b c h w -> b h w c'),
#             nn.Linear(in_channels, dim),
#             Rearrange('b h w c -> b (h w) c')
#         )
        
#         self.pos_embed_mlp = nn.Sequential(
#             nn.Linear(2, dim//2),
#             nn.GELU(),
#             nn.Linear(dim//2, dim)
#         )
        
#         self.register_buffer('band_freq', 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)))
        
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.transformer = nn.TransformerEncoder(
#             encoder_layer=nn.TransformerEncoderLayer(
#                 d_model=dim,
#                 nhead=heads,
#                 dim_feedforward=dim*4,
#                 dropout=dropout,
#                 activation='gelu',
#                 batch_first=True,
#                 norm_first=True
#             ),
#             num_layers=depth
#         )
#         self.reg_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, 1),
#             nn.ReLU()
#         )

#     def get_position_embedding(self, h, w, device):
#         y_coords = (torch.arange(h, device=device).float() + 0.5) / h
#         x_coords = (torch.arange(w, device=device).float() + 0.5) / w
#         grid = torch.stack(torch.meshgrid(x_coords, y_coords, indexing='xy'), dim=-1)  # [H, W, 2]
        
#         pos_embed = self.pos_embed_mlp(grid)  # [H, W, dim]
#         return pos_embed.view(1, h*w, self.dim)  # [1, N_patches, dim]

#     def get_band_embedding(self, band_idx, total_bands, device):
#         # last band index is 0
#         pos = total_bands - 1 - band_idx
        
#         angles = pos * self.band_freq  # [dim//2]
        
#         emb = torch.zeros(1, 1, self.dim, device=device)
#         emb[0, 0, 0::2] = torch.sin(angles)
#         emb[0, 0, 1::2] = torch.cos(angles)
        
#         return emb  # [1, 1, dim]
    
#     def forward(self, band_features):
#         """
#         band_features: list [B, H_i, W_i, C_i]
#         """
#         all_patches = []
        
#         total_bands = len(band_features)
#         for band_idx, feat in enumerate(band_features):
#             B, H, W, C = feat.shape
        
#             patches = self.patch_embed(feat)  # [B, N_patches, dim]
            
#             pos_embed = self.get_position_embedding(H, W, feat.device)
#             patches += pos_embed
            
#             band_embed = self.get_band_embedding(band_idx, total_bands, feat.device)
#             patches += band_embed
            
#             all_patches.append(patches)
        
#         del band_features

#         x = torch.cat(all_patches, dim=1)  # [B, total_patches, dim]

#         del all_patches
        
#         cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)

#         x = self.transformer(x)
        
#         cls_feat = x[:, 0]

#         return self.reg_head(cls_feat).squeeze(-1)

# class cvvdp_ml_transformer_bands(cvvdp_ml_base):
#     def __init__(self,
#                  dim=256,
#                  **kwargs):
        
#         self.set_device( kwargs.get('device') )
        
#         self.transformer_net = RegressionTransformer_bands(
#             dim=dim
#         ).to(self.device)

#         super().__init__(**kwargs)

#     def get_nets_to_load(self):
#         return ['transformer_net']
    
#     def do_pooling_and_jods(self, features):
#         Q_JOD = torch.as_tensor(10., device=self.device)
#         is_image = (features[0].shape[3]==3)
#         no_bands = len(features)

#         input_features = []
#         for bb, f in enumerate(features):
#             f[..., 1::2] = torch.sqrt(torch.abs(f[..., 1::2]))

#             if is_image:
#                 f = torch.cat( (f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3)
#             if self.disabled_features is not None:
#                 f[..., self.disabled_features] = 0

#             # band_features = [
#             #     f[..., 0:4].flatten(start_dim=3),
#             #     f[..., 4:].flatten(start_dim=3)
#             # ]
            
#             # f_all = torch.cat([
#             #     f[..., 0:4].flatten(start_dim=3),
#             #     f[..., 4:].flatten(start_dim=3)
#             # ], dim=-1)

#             f = f.flatten(start_dim=3)

#             #band_features = f_all.permute(0, 3, 1, 2)  # [B, C_i, H_i, W_i]

#             input_features.append(f)
        
#         del features 

#         delta = self.transformer_net(input_features) / no_bands

#         if is_image:
#             delta *= self.image_int

#         Q_JOD -= delta.mean()

#         return Q_JOD

# register_metric( cvvdp_ml_transformer_bands )
