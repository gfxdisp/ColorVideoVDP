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
import timm
from pathlib import Path
import importlib.util
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

from pycvvdp.cvvdp_metric import cvvdp, safe_pow, cvvdp_frame_buffers
from pycvvdp.vq_metric import vq_exception

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

        if self.heatmap is not None and self.heatmap!='none' and not getattr(self, 'supports_heatmap', False):
            raise vq_exception( "Currently cvvdp-ml metrics do not produce heatmaps" )

        self.train(False)

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

    def _rewind_video_source_if_supported(self, vid_source):
        rewind_fn = getattr(vid_source, 'rewind', None)
        if callable(rewind_fn):
            rewind_fn()


    # Switch to training mode (e.g., to optimize memory allocation)
    def train(self, do_training=True):
        super().train(do_training)
        for net in self.get_nets_to_load():
            for param in getattr(self, net).parameters():
                param.requires_grad = do_training
            getattr(self, net).train(do_training)
            # if not do_training:            
            #     for param in getattr(self, net).parameters():
            #         param.requires_grad = False    

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

        no_channels = 2+temp_ch

        if self.do_heatmap:
            dmap_channels = 1 if self.heatmap == "raw" else 3
            heatmap = torch.zeros([1,dmap_channels,N_frames,height,width], dtype=torch.float16, device=torch.device('cpu')) # Store heatmap in the CPU memory
        else:
            heatmap = None

        # Q_per_ch = None

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
        # feature_size = math.floor(self.pix_per_deg) 

        features = None

        fb = cvvdp_frame_buffers()

        for ff in range(0, N_frames, block_N_frames):
            cur_block_N_frames = min(block_N_frames,N_frames-ff) # How many frames in this block?

            R = self.read_block_of_frames(vid_source, no_channels, fb, block_N_frames, met_colorspace, ff, cur_block_N_frames)

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

            ff_end = ff+features_per_block[bb].shape[1]
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

    def export_distogram(self, stats, fname, jod_max=None, base_size=6):
        raise vq_exception( 'Currently cvvdp-ml metrics do not export distograms')



"""
ColorVideoVDP metric with an MLP head.
"""
class cvvdp_ml(cvvdp_ml_base):

    # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
    # random_init - do not load NN from a checkpoint file, use a random initialization
    def __init__(self, device=None, **kwargs):

        self.set_device( device )

        # self.feature_net = nn.Sequential(
        #     nn.Linear(8, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, 1),
        # ).to(self.device)

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

    def _pool_temporal_delta(self, delta, batch, frames):
        delta = delta.reshape(batch, frames)
        return delta.mean(dim=-1).mean()

    # Perform pooling with per-band weights and map to JODs
    def do_pooling_and_jods(self, features):
        # features[band][frames,width,height,channels,stat]
        # disables_features is an array of indices of the stat to be disabled

        no_bands = len(features)

        q_jod = torch.as_tensor(10., device=self.device)

        f0 = features[0]
        ch_dim = 4 if f0.dim() == 6 else 3
        is_image = (f0.shape[ch_dim] == 3)

        band_scores = []

        for bb in range(no_bands):
            f = features[bb]

            if is_image:
                if f.dim() == 6:
                    f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], f.shape[3], 1, f.shape[5]), device=self.device)), dim=4)
                else:
                    f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3)

            if self.disabled_features is not None:
                if f.dim() == 6:
                    f[..., self.disabled_features] = 0
                else:
                    f[:, :, :, :, self.disabled_features] = 0

            f_d = f[..., 4:]
            f_d[..., 1] = torch.sqrt(torch.abs(f_d[..., 1]))
            f_d = f_d.flatten(start_dim=f_d.dim() - 2)

            if f_d.dim() == 5:
                batch, frames, height, width, _ = f_d.shape
                f_d_flat = f_d.reshape(-1, 8)
                token_scores = self.feature_net(f_d_flat).reshape(batch, frames, height * width)
                scores = token_scores.mean(dim=2)

            elif f_d.dim() == 4:
                batch, height, width, _ = f_d.shape
                frames = 1
                f_d_flat = f_d.reshape(-1, 8)
                token_scores = self.feature_net(f_d_flat).reshape(batch, height * width)
                scores = token_scores.mean(dim=1, keepdim=True)
            else:
                raise RuntimeError(f"Unsupported feature tensor dimensionality: {f_d.dim()}")

            if bb == no_bands - 1:
                scores = scores * self.baseband_weight

            band_scores.append(scores)

        band_scores = torch.stack(band_scores, dim=2)
        delta = band_scores.mean(dim=2)
        batch, frames = delta.shape[0], delta.shape[1]
        delta = self._pool_temporal_delta(delta, batch, frames)

        if is_image:
            delta *= self.image_int

        q_jod = q_jod - delta

        assert(not q_jod.isnan())
        return q_jod

register_metric(cvvdp_ml)


class cvvdp_ml_temporal_hierarchical_topk(cvvdp_ml):
    """
    cvvdp_ml variant with hierarchical temporal pooling.

    Pooling strategy:
    - Split the sequence into coarse windows
    - Inside each coarse window, split into fine windows
    - Use a percentile inside each fine window to pick the worse frames in that segment
    - Select the worst fine windows per coarse window (top-k)
    - Fuse coarse mean with the fine worst summary using alpha
    - Average across coarse windows and the batch
    """

    def __init__(self,
                 device=None,
                 coarse_window_s=1.0,
                 coarse_overlap=0.5,
                 fine_window_s=0.1,
                 fine_overlap=0.5,
                 fine_percentile=0.8,
                 topk_worst=1,
                 topk_mode='mean',
                 topk_softmax_temperature=10.0,
                 coarse_fine_fusion_alpha=0.5,
                 temporal_fps_hint=None,
                 **kwargs):
        self.coarse_window_s = float(coarse_window_s)
        self.coarse_overlap = float(coarse_overlap)
        self.fine_window_s = float(fine_window_s)
        self.fine_overlap = float(fine_overlap)
        self.fine_percentile = float(fine_percentile)
        self.topk_worst = int(topk_worst)
        self.topk_mode = str(topk_mode)
        self.topk_softmax_temperature = float(topk_softmax_temperature)
        self.coarse_fine_fusion_alpha = float(coarse_fine_fusion_alpha)
        self.temporal_fps_hint = temporal_fps_hint
        self._temporal_pool_fps = None
        super().__init__(device=device, **kwargs)

    def predict_video_source(self, vid_source):
        try:
            fps = float(vid_source.get_frames_per_second())
            if not math.isfinite(fps) or fps <= 0:
                fps = None
        except Exception:
            fps = None

        self._temporal_pool_fps = fps
        try:
            return super().predict_video_source(vid_source)
        finally:
            self._temporal_pool_fps = None

    def _resolve_temporal_pool_fps(self):
        candidates = [
            self._temporal_pool_fps,
            self.temporal_fps_hint,
            getattr(self, 'fps', None),
            getattr(self, 'frames_per_second', None),
            getattr(self, 'source_fps', None),
            getattr(self, 'nominal_fps', None),
        ]

        for cand in candidates:
            try:
                fps = float(cand)
                if math.isfinite(fps) and fps > 0:
                    return fps
            except Exception:
                continue

        return 30.0

    def _reduce_worst_fine_values(self, fine_values):
        if fine_values.shape[-1] <= 1:
            return fine_values.squeeze(-1)

        if self.topk_mode == 'max':
            return fine_values.max(dim=-1).values

        if self.topk_mode == 'mean':
            return fine_values.mean(dim=-1)

        if self.topk_mode == 'softmax':
            tau = torch.as_tensor(self.topk_softmax_temperature, device=fine_values.device, dtype=fine_values.dtype)
            if not torch.isfinite(tau) or tau <= 0:
                return fine_values.mean(dim=-1)

            ref_vals = fine_values.max(dim=-1, keepdim=True).values
            logits = (fine_values - ref_vals) * tau
            weights = torch.softmax(logits, dim=-1)
            return (weights * fine_values).sum(dim=-1)

        raise ValueError(f"Unsupported topk_mode: {self.topk_mode}. Expected one of ['max', 'mean', 'softmax']")

    def _pool_temporal_delta(self, delta, batch, frames):
        delta = delta.reshape(batch, frames)
        if frames <= 1:
            return delta.mean(dim=-1).mean()

        fps = self._resolve_temporal_pool_fps()
        coarse_window_frames = max(1, int(round(self.coarse_window_s * fps)))
        fine_window_frames = max(1, int(round(self.fine_window_s * fps)))

        coarse_overlap = max(0.0, min(0.95, self.coarse_overlap))
        fine_overlap = max(0.0, min(0.95, self.fine_overlap))
        coarse_hop = max(1, int(round(coarse_window_frames * (1.0 - coarse_overlap))))
        fine_hop = max(1, int(round(fine_window_frames * (1.0 - fine_overlap))))

        k_worst = max(1, self.topk_worst)
        fusion_alpha = max(0.0, min(1.0, self.coarse_fine_fusion_alpha))
        fusion_alpha = torch.as_tensor(fusion_alpha, dtype=delta.dtype, device=delta.device)
        fine_percentile = max(0.0, min(1.0, self.fine_percentile))

        coarse_scores = []

        for coarse_start in range(0, frames, coarse_hop):
            coarse_end = min(coarse_start + coarse_window_frames, frames)
            coarse_seg = delta[:, coarse_start:coarse_end]
            if coarse_seg.shape[1] == 0:
                continue

            coarse_mean = coarse_seg.mean(dim=-1)

            fine_seg_values = []
            for fine_start in range(coarse_start, coarse_end, fine_hop):
                fine_end = min(fine_start + fine_window_frames, coarse_end)
                fine_seg = delta[:, fine_start:fine_end]
                if fine_seg.shape[1] == 0:
                    continue
                fine_seg_values.append(torch.quantile(fine_seg, q=fine_percentile, dim=-1))

            if len(fine_seg_values) == 0:
                adjusted_coarse = coarse_mean
            else:
                fine_seg_values = torch.stack(fine_seg_values, dim=-1)
                k = min(k_worst, fine_seg_values.shape[-1])
                worst_k = torch.topk(fine_seg_values, k=k, dim=-1, largest=True).values
                worst_summary = self._reduce_worst_fine_values(worst_k)
                adjusted_coarse = (1.0 - fusion_alpha) * coarse_mean + fusion_alpha * worst_summary

            coarse_scores.append(adjusted_coarse)

        if len(coarse_scores) == 0:
            return delta.mean(dim=-1).mean()

        return torch.stack(coarse_scores, dim=-1).mean(dim=-1).mean()


register_metric(cvvdp_ml_temporal_hierarchical_topk)


class cvvdp_ml_temporal_multiscale_pooling(cvvdp_ml):
    """
    cvvdp_ml variant with multiscale temporal pooling.

    Pooling strategy:
    - Build overlapping temporal windows at multiple scales
    - For each scale, compute per-segment statistics (mean and percentile)
    - Select a soft-worst segment score for each statistic
    - Average across the categories and then across the batch
    """

    def __init__(self,
                 device=None,
                 temporal_windows_s=(0.05, 1.0),
                 temporal_percentile=0.2,
                 temporal_overlap=0.5,
                 soft_worst_temperature=10.0,
                 temporal_use_mean=True,
                 temporal_use_percentile=False,
                 temporal_fps_hint=None,
                 **kwargs):
        self.temporal_windows_s = tuple(float(w) for w in temporal_windows_s)
        self.temporal_percentile = float(temporal_percentile)
        self.temporal_overlap = float(temporal_overlap)
        self.soft_worst_temperature = float(soft_worst_temperature)
        self.temporal_use_mean = bool(temporal_use_mean)
        self.temporal_use_percentile = bool(temporal_use_percentile)
        self.temporal_fps_hint = temporal_fps_hint
        self._temporal_pool_fps = None
        super().__init__(device=device, **kwargs)

    def predict_video_source(self, vid_source):
        try:
            fps = float(vid_source.get_frames_per_second())
            if not math.isfinite(fps) or fps <= 0:
                fps = None
        except Exception:
            fps = None

        self._temporal_pool_fps = fps
        try:
            return super().predict_video_source(vid_source)
        finally:
            self._temporal_pool_fps = None

    def _soft_worst(self, values, dim=-1, mode='max'):
        if values.shape[dim] <= 1:
            return values.squeeze(dim)

        tau = torch.as_tensor(self.soft_worst_temperature, device=values.device, dtype=values.dtype)
        if not torch.isfinite(tau) or tau <= 0:
            return values.max(dim=dim).values if mode == 'max' else values.min(dim=dim).values

        if mode == 'max':
            ref_vals = values.max(dim=dim, keepdim=True).values
            logits = (values - ref_vals) * tau
        elif mode == 'min':
            ref_vals = values.min(dim=dim, keepdim=True).values
            logits = -(values - ref_vals) * tau
        else:
            raise ValueError(f"Unsupported soft-worst mode: {mode}")

        weights = torch.softmax(logits, dim=dim)
        return (weights * values).sum(dim=dim)

    def _resolve_temporal_pool_fps(self):
        candidates = [
            self._temporal_pool_fps,
            self.temporal_fps_hint,
            getattr(self, 'fps', None),
            getattr(self, 'frames_per_second', None),
            getattr(self, 'source_fps', None),
            getattr(self, 'nominal_fps', None),
        ]

        for cand in candidates:
            try:
                fps = float(cand)
                if math.isfinite(fps) and fps > 0:
                    return fps
            except Exception:
                continue

        return 30.0

    def _pool_temporal_delta(self, delta, batch, frames):
        delta = delta.reshape(batch, frames)
        if frames <= 1:
            return delta.mean(dim=-1).mean()

        use_mean = self.temporal_use_mean
        use_percentile = self.temporal_use_percentile
        if not use_mean and not use_percentile:
            use_mean = True

        p = max(0.0, min(1.0, self.temporal_percentile))
        overlap = max(0.0, min(0.95, self.temporal_overlap))
        fps = self._resolve_temporal_pool_fps()

        category_values = []

        for window_s in self.temporal_windows_s:
            window_frames = max(1, int(round(window_s * fps)))
            hop_frames = max(1, int(round(window_frames * (1.0 - overlap))))

            seg_mean_vals = []
            seg_perc_vals = []

            for start in range(0, frames, hop_frames):
                end = min(start + window_frames, frames)
                seg = delta[:, start:end]
                if seg.shape[1] == 0:
                    continue

                if use_mean:
                    seg_mean_vals.append(seg.mean(dim=-1))

                if use_percentile:
                    seg_perc_vals.append(torch.quantile(seg, q=p, dim=-1))

            if use_mean and len(seg_mean_vals) > 0:
                seg_mean = torch.stack(seg_mean_vals, dim=-1)
                seg_mean_quality = 10.0 - seg_mean
                worst_mean_quality = self._soft_worst(seg_mean_quality, dim=-1, mode='min')
                worst_mean_delta = 10.0 - worst_mean_quality
                category_values.append(worst_mean_delta)

            if use_percentile and len(seg_perc_vals) > 0:
                seg_perc = torch.stack(seg_perc_vals, dim=-1)
                seg_perc_quality = 10.0 - seg_perc
                worst_perc_quality = self._soft_worst(seg_perc_quality, dim=-1, mode='min')
                worst_perc_delta = 10.0 - worst_perc_quality
                category_values.append(worst_perc_delta)

        if len(category_values) == 0:
            return delta.mean(dim=-1).mean()

        pooled = torch.stack(category_values, dim=-1).mean(dim=-1)
        return pooled.mean()


register_metric(cvvdp_ml_temporal_multiscale_pooling)

class cvvdp_ml_freq_bands(cvvdp_ml_base):
    """
    ColorVideoVDP-ML variant that appends per-band spatial frequency to patch features.

    This follows the key idea of the band-frequency DINO-fusion variant, but it uses
    a single feature net that predicts quality directly and does not use baseband weight.
    """

    def __init__(self, device=None, band_freqs=None, **kwargs):

        self.set_device(device)

        self.feature_input_dim = 9  # 8 D features + 1 band-frequency scalar
        self.feature_net = nn.Sequential(
            nn.Linear(self.feature_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        ).to(self.device)

        self.band_freqs = band_freqs

        super().__init__(device=device, **kwargs)

    def get_nets_to_load(self):
        return ['feature_net']

    def _append_band_frequency(self, f_d, band_freq):
        band_freq = torch.as_tensor(band_freq, device=self.device, dtype=f_d.dtype)
        freq_map = torch.ones(f_d.shape[:-1] + (1,), device=self.device, dtype=f_d.dtype) * band_freq
        return torch.cat((f_d, freq_map), dim=-1)

    def _resolve_band_freqs(self, no_bands):
        if self.band_freqs is not None:
            band_freqs = torch.as_tensor(self.band_freqs, device=self.device, dtype=torch.float32).flatten()
            if band_freqs.numel() == 1:
                band_freqs = band_freqs.expand(no_bands)
            elif band_freqs.numel() < no_bands:
                tail = band_freqs[-1:].expand(no_bands - band_freqs.numel())
                band_freqs = torch.cat((band_freqs, tail), dim=0)
            else:
                band_freqs = band_freqs[:no_bands]
            return band_freqs

        ppd = getattr(self, 'pix_per_deg', None)
        if ppd is not None:
            ppd = float(ppd)
        if ppd is not None and np.isfinite(ppd) and ppd > 0:
            height = max(int(no_bands) - 1, 0)
            band_freqs = np.array([1.0] + [0.3228 * 2.0 ** (-f) for f in range(height)], dtype=np.float32) * (ppd / 2.0)
            if band_freqs.shape[0] < no_bands:
                tail = np.repeat(band_freqs[-1:], no_bands - band_freqs.shape[0])
                band_freqs = np.concatenate([band_freqs, tail], axis=0)
            return torch.as_tensor(band_freqs[:no_bands], device=self.device)

        if getattr(self, 'lpyr', None) is not None:
            band_freqs = torch.as_tensor(self.lpyr.get_freqs(), device=self.device, dtype=torch.float32).flatten()
            if band_freqs.numel() > 0:
                if band_freqs.numel() >= no_bands:
                    return band_freqs[:no_bands]
                tail = band_freqs[-1:].expand(no_bands - band_freqs.numel())
                return torch.cat((band_freqs, tail), dim=0)

        ppd = 1.0
        height = max(int(no_bands) - 1, 0)
        band_freqs = np.array([1.0] + [0.3228 * 2.0 ** (-f) for f in range(height)], dtype=np.float32) * (ppd / 2.0)
        if band_freqs.shape[0] < no_bands:
            tail = np.repeat(band_freqs[-1:], no_bands - band_freqs.shape[0])
            band_freqs = np.concatenate([band_freqs, tail], axis=0)
        return torch.as_tensor(band_freqs[:no_bands], device=self.device)

    def do_pooling_and_jods(self, features):
        no_bands = len(features)
        q_jod = torch.as_tensor(10., device=self.device)

        f0 = features[0]
        ch_dim = 4 if f0.dim() == 6 else 3
        is_image = (f0.shape[ch_dim] == 3)

        rho_band = self._resolve_band_freqs(no_bands)

        band_scores = []

        for bb in range(no_bands):
            f = features[bb]

            if is_image:
                if f.dim() == 6:
                    f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], f.shape[3], 1, f.shape[5]), device=self.device)), dim=4)
                else:
                    f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3)

            if self.disabled_features is not None:
                if f.dim() == 6:
                    f[..., self.disabled_features] = 0
                else:
                    f[:, :, :, :, self.disabled_features] = 0

            f_d = f[..., 4:]
            f_d[..., 1] = torch.sqrt(torch.abs(f_d[..., 1]))
            f_d = f_d.flatten(start_dim=f_d.dim() - 2)
            f_d = self._append_band_frequency(f_d, rho_band[bb])

            if f_d.dim() == 5:
                batch, frames, height, width, feat_dim = f_d.shape
                f_d_flat = f_d.reshape(-1, feat_dim)
                token_scores = self.feature_net(f_d_flat).reshape(batch, frames, height * width)
                scores = token_scores.mean(dim=2)

            elif f_d.dim() == 4:
                batch, height, width, feat_dim = f_d.shape
                frames = 1
                f_d_flat = f_d.reshape(-1, feat_dim)
                token_scores = self.feature_net(f_d_flat).reshape(batch, height * width)
                scores = token_scores.mean(dim=1, keepdim=True)
            else:
                raise RuntimeError(f"Unsupported feature tensor dimensionality: {f_d.dim()}")

            band_scores.append(scores)

        band_scores = torch.stack(band_scores, dim=2)
        delta = band_scores.mean(dim=2)
        delta = delta.mean(dim=-1).mean()

        if is_image:
            delta *= self.image_int

        q_jod = q_jod - delta

        assert(not q_jod.isnan())
        return q_jod


register_metric(cvvdp_ml_freq_bands)


class cvvdp_ml_dino_base(cvvdp_ml_base):

    def __init__(self, dino_net='dino_v1', dino_token='cls', device=None, **kwargs):

        self.set_device(device)

        super().__init__(device=device, **kwargs)

        self.dino_net = dino_net
        self.dino_token = dino_token
        self.dino_colorspace = 'display_encoded_01'

        if self.dino_net != 'dino_v1':
            raise ValueError(f"Unsupported DINO backbone: {self.dino_net}. Only 'dino_v1' is currently supported.")

        self.dino = self._create_timm_model([
            "vit_base_patch16_224.dino",
        ])

        self.dino_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        
    def predict_video_source(self, vid_source):
        assert vid_source.get_batch_size()==1 or self.heatmap is None or self.heatmap=='none', 'Heatmaps not supported when batches are used'

        features, heatmap = self.extract_features(vid_source)
        dino_features = self.extract_features_dino(vid_source)
        Q_jod = self.do_pooling_and_jods(features, dino_features)

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

    @abstractmethod
    def do_pooling_and_jods(self, features, dino_features):
        """
        """

    def _create_timm_model(self, candidates):
        errors = []
        for model_name in candidates:
            try:
                return timm.create_model(model_name, pretrained=True).to(self.device).eval()
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")

        raise RuntimeError(
            f"None of the requested timm models could be loaded: {candidates}. "
            f"Errors: {' ; '.join(errors)}"
        )

    @staticmethod
    def _forward_tokens(model, patches):
        feats = model.forward_features(patches)

        if isinstance(feats, dict):
            cls_tokens = feats.get("x_norm_clstoken")
            patch_tokens = feats.get("x_norm_patchtokens")

            if cls_tokens is not None and patch_tokens is not None:
                return torch.cat([cls_tokens.unsqueeze(1), patch_tokens], dim=1)

            x_norm = feats.get("x_norm")
            if isinstance(x_norm, torch.Tensor) and x_norm.ndim == 3:
                return x_norm

            for value in feats.values():
                if isinstance(value, torch.Tensor) and value.ndim == 3:
                    return value

            raise RuntimeError("Could not extract token tensor from timm forward_features dict output.")

        if isinstance(feats, (list, tuple)):
            for value in feats:
                if isinstance(value, torch.Tensor) and value.ndim == 3:
                    return value
            raise RuntimeError("Could not extract token tensor from timm forward_features sequence output.")

        if isinstance(feats, torch.Tensor) and feats.ndim == 3:
            return feats

        raise RuntimeError("Unsupported forward_features output format for timm DINO model.")

    @staticmethod
    @torch.no_grad()
    def tokens_dino(model, patches, token='cls'):
        tokens = cvvdp_ml_dino_base._forward_tokens(model, patches)

        if token == 'cls':
            return tokens[:, 0]
        elif token == 'patch':
            return tokens[:, 1:]
        else:
            raise ValueError(f"Unsupported token type: {token}")

    @torch.no_grad()
    def extract_features_dino(self, vid_source):
        _, _, N_frames = vid_source.get_video_size()

        self._rewind_video_source_if_supported(vid_source)

        ref_features = []

        for ff in range(N_frames):
            ref_frame = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.dino_colorspace).squeeze(2).clamp(min=0, max=1)

            ref_frame = Func.interpolate(ref_frame, size=(224, 224), mode='bilinear', align_corners=False)

            ref_frame = (ref_frame - self.dino_mean) / self.dino_std

            ref_cls = self.tokens_dino(self.dino, ref_frame, token=self.dino_token)

            ref_features.append(ref_cls)

        ref_features = torch.stack(ref_features, dim=1)

        return [ref_features]


class cvvdp_ml_saliency_base(cvvdp_ml_base):

    def __init__(self, device=None, stsanet_weights=None, stsanet_resolution=(224, 384), stsanet_clip_len=32, **kwargs):

        self.set_device(device)

        super().__init__(device=device, **kwargs)

        self.requires_saliency_features = True
        self.stsanet_clip_len = int(stsanet_clip_len)
        self.stsanet_left_context = self.stsanet_clip_len // 2 - 1
        self.stsanet_right_context = self.stsanet_clip_len // 2
        self.stsanet_resolution = tuple(stsanet_resolution)
        self.saliency_colorspace = 'display_encoded_01'
        self.stsanet_weights = stsanet_weights

        self.stsanet = self._create_stsanet_model().to(self.device).eval()
        for param in self.stsanet.parameters():
            param.requires_grad = False

    def predict_video_source(self, vid_source):
        assert vid_source.get_batch_size()==1 or self.heatmap is None or self.heatmap=='none', 'Heatmaps not supported when batches are used'

        features, heatmap = self.extract_features(vid_source)
        saliency_features = self.extract_features_saliency(vid_source)
        Q_jod = self.do_pooling_and_jods(features, saliency_features)

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

    @abstractmethod
    def do_pooling_and_jods(self, features, saliency_features):
        """
        """

    def _get_project_root(self):
        return Path(__file__).resolve().parents[2]

    def _get_stsanet_weights_path(self):
        if self.stsanet_weights is not None and self.stsanet_weights != '':
            return Path(self.stsanet_weights)
        # return self._get_project_root() / 'external' / 'STSANet' / 'STSANet_fine-tuned_on_UCF.pth'
        # return self._get_project_root() / 'external' / 'STSANet' / 'STSANet_DHF1K.pth'
        # return self._get_project_root() / 'external' / 'STSANet' / 'STSANet_fine-tuned_on_DIEM.pth'
        return self._get_project_root() / 'external' / 'STSANet' / 'STSANet_fine-tuned_on_Hollywood.pth'

    def _import_stsanet_class(self):
        stsa_model_path = self._get_project_root() / 'external' / 'STSANet' / 'STSA_model.py'
        if not stsa_model_path.exists():
            raise RuntimeError(f'STSANet model file not found: {stsa_model_path}')

        module_name = 'external_stsanet_module'
        spec = importlib.util.spec_from_file_location(module_name, str(stsa_model_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Could not load STSANet module from: {stsa_model_path}')

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, 'STSANet'):
            raise RuntimeError(f'Module {stsa_model_path} does not expose STSANet class.')

        return module.STSANet

    def _create_stsanet_model(self):
        stsanet_cls = self._import_stsanet_class()
        model = stsanet_cls()

        weights_path = self._get_stsanet_weights_path()
        if not weights_path.exists():
            logging.warning(f'STSANet weights not found at {weights_path}. Using random initialization.')
            return model

        checkpoint = torch.load(str(weights_path), map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'Unexpected STSANet checkpoint format at {weights_path}: expected a state_dict-like mapping.')

        model_keys = set(model.state_dict().keys())

        candidates = [('raw', checkpoint)]
        remapped = {k.split('.', 1)[1]: v for k, v in checkpoint.items() if isinstance(k, str) and '.' in k}
        if len(remapped) > 0:
            candidates.append(('stripped_prefix', remapped))

        best_name = None
        best_state = None
        best_overlap = -1
        for cand_name, cand_state in candidates:
            overlap = len(model_keys.intersection(cand_state.keys()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_name = cand_name
                best_state = cand_state

        if best_overlap <= 0:
            raise RuntimeError(
                f'Could not match STSANet checkpoint keys to model keys for {weights_path}. '
                f'Please verify that weights correspond to external/STSANet/STSA_model.py.'
            )

        incompatible = model.load_state_dict(best_state, strict=False)
        matched_params = len(model_keys) - len(incompatible.missing_keys)
        if matched_params <= 0:
            raise RuntimeError(
                f'STSANet checkpoint load resulted in 0 matched parameters from {weights_path}. '
                f'Chosen mapping: {best_name}.'
            )

        logging.info(
            f'Loaded STSANet weights from: {weights_path} using {best_name} mapping '
            f'({matched_params}/{len(model_keys)} params matched).'
        )
        return model

    def _build_stsanet_clip(self, ref_video, center_index):
        total_frames = ref_video.shape[0]
        clip_indices = [
            min(max(center_index - self.stsanet_left_context + offset, 0), total_frames - 1)
            for offset in range(self.stsanet_clip_len)
        ]
        clip = ref_video[clip_indices, :, :, :]
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0)
        return clip

    @torch.no_grad()
    def extract_features_saliency(self, vid_source):
        _, _, N_frames = vid_source.get_video_size()
        target_h, target_w = self.stsanet_resolution

        self._rewind_video_source_if_supported(vid_source)

        if N_frames <= 0:
            return [torch.empty((1, 0, target_h, target_w), dtype=torch.float32, device=self.device)]

        ref_frames = []
        for ff in range(N_frames):
            ref_frame = vid_source.get_reference_frame(ff, device=self.device, colorspace=self.saliency_colorspace).squeeze(2).clamp(min=0, max=1)
            ref_frame = Func.interpolate(ref_frame, size=(target_h, target_w), mode='bilinear', align_corners=False)
            ref_frames.append(ref_frame.squeeze(0))

        ref_video = torch.stack(ref_frames, dim=0)

        saliency_maps = []
        self.stsanet.eval()
        for ff in range(N_frames):
            clip = self._build_stsanet_clip(ref_video, ff)
            sal_map = self.stsanet(clip).squeeze(0)
            saliency_maps.append(sal_map)

        saliency_maps = torch.stack(saliency_maps, dim=0).unsqueeze(0)
        return [saliency_maps]


# class cvvdp_ml_dino(cvvdp_ml_dino_base):

#     def __init__(self, device=None, fusion_mode='concat_fused', hidden_dim=128, **kwargs):
#         self.set_device(device)
#         ch_no = 4
#         stats_no = 2
#         dino_dim = 768

#         self.fusion_mode = fusion_mode
#         self.hidden_dim = int(hidden_dim)

#         if self.fusion_mode not in ['concat', 'concat_fused', 'add', 'film']:
#             raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}. Expected one of ['concat', 'concat_fused', 'add', 'film']")

#         if self.fusion_mode == 'concat':
#             mlp_in_channels = stats_no * ch_no + dino_dim
#             self.feature_net = MLP(
#                 in_channels=mlp_in_channels,
#                 hidden_channels=[256, 128, 1],
#                 activation_layer=torch.nn.ReLU,
#                 dropout=0.2,
#             ).to(self.device)
#         elif self.fusion_mode == 'concat_fused':
#             if 'random_init' not in kwargs:
#                 kwargs['random_init'] = True

#             feat_dim = stats_no * ch_no
#             self.patch_proj = nn.Linear(feat_dim, 256, bias=True).to(self.device)
#             self.dino_proj = nn.Linear(dino_dim, 256, bias=False).to(self.device)
#             self.feature_net = nn.Sequential(
#                 nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(256, 128),
#                 nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(128, 1),
#             ).to(self.device)
#         else:
#             if 'random_init' not in kwargs:
#                 kwargs['random_init'] = True

#             feat_dim = stats_no * ch_no
#             self.patch_proj = nn.Sequential(
#                 nn.LayerNorm(feat_dim),
#                 nn.Linear(feat_dim, self.hidden_dim),
#                 nn.ReLU(),
#             ).to(self.device)
#             self.dino_proj = nn.Sequential(
#                 nn.LayerNorm(dino_dim),
#                 nn.Linear(dino_dim, self.hidden_dim),
#             ).to(self.device)

#             if self.fusion_mode == 'film':
#                 self.film_scale = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
#                 self.film_bias = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)

#             self.feature_net = nn.Sequential(
#                 nn.LayerNorm(self.hidden_dim),
#                 nn.Linear(self.hidden_dim, max(32, self.hidden_dim // 2)),
#                 nn.GELU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(max(32, self.hidden_dim // 2), 1),
#             ).to(self.device)

#         super().__init__(device=device, **kwargs)

#         self.train(False)

#     def get_nets_to_load(self):
#         if self.fusion_mode == 'concat':
#             return ['feature_net'] if hasattr(self, 'feature_net') else []

#         nets = ['patch_proj', 'dino_proj', 'feature_net']
#         if self.fusion_mode == 'film':
#             nets.extend(['film_scale', 'film_bias'])
#         return nets

#     def _align_dino(self, dino_ref, batch, frames):
#         if dino_ref.dim() == 2:
#             dino_ref = dino_ref.unsqueeze(0)

#         if dino_ref.shape[0] == 1 and batch > 1:
#             dino_ref = dino_ref.expand(batch, -1, -1)

#         if dino_ref.shape[1] != frames:
#             if dino_ref.shape[1] == 1:
#                 dino_ref = dino_ref.expand(-1, frames, -1)
#             else:
#                 min_frames = min(dino_ref.shape[1], frames)
#                 dino_ref = dino_ref[:, :min_frames, :]
#                 if min_frames < frames:
#                     tail = dino_ref[:, -1:, :].expand(-1, frames - min_frames, -1)
#                     dino_ref = torch.cat((dino_ref, tail), dim=1)

#         return dino_ref

#     def do_pooling_and_jods(self, features, dino_features):
#         no_bands = len(features)
#         q_jod = torch.as_tensor(10., device=self.device)

#         f0 = features[0]
#         ch_dim = 4 if f0.dim() == 6 else 3
#         is_image = (f0.shape[ch_dim] == 3)

#         dino_ref = dino_features[0]
#         if dino_ref.dim() == 2:
#             dino_ref = dino_ref.unsqueeze(0)
#         dino_ref = dino_ref.to(self.device)

#         for bb in range(no_bands):
#             f = features[bb]

#             if is_image:
#                 if f.dim() == 6:
#                     f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], f.shape[3], 1, f.shape[5]), device=self.device)), dim=4)
#                 else:
#                     f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3)

#             if self.disabled_features is not None:
#                 if f.dim() == 6:
#                     f[..., self.disabled_features] = 0
#                 else:
#                     f[:, :, :, :, self.disabled_features] = 0

#             f_d = f[..., 4:]
#             f_d[..., 1] = torch.sqrt(torch.abs(f_d[..., 1]))

#             f_d = f_d.flatten(start_dim=f_d.dim() - 2)

#             if f_d.dim() != 5:
#                 raise RuntimeError(f"Expected 5D f_d tensor [B,F,H,W,C], got shape {tuple(f_d.shape)}")

#             batch, frames, height, width, feat_dim = f_d.shape
#             dino_ref_cur = self._align_dino(dino_ref, batch, frames)

#             if self.fusion_mode == 'concat':
#                 dino_map = dino_ref_cur[:, :, None, None, :].expand(-1, -1, height, width, -1)
#                 f_cat = torch.cat((f_d, dino_map), dim=-1)
#                 d_all = self.feature_net(f_cat)
#             elif self.fusion_mode == 'concat_fused':
#                 patch_latent = self.patch_proj(f_d)
#                 dino_latent = self.dino_proj(dino_ref_cur)
#                 fused = patch_latent + dino_latent[:, :, None, None, :]
#                 d_all = self.feature_net(fused)
#             else:
#                 patch_latent = self.patch_proj(f_d.reshape(-1, feat_dim)).reshape(batch, frames, height, width, self.hidden_dim)
#                 dino_latent = self.dino_proj(dino_ref_cur)

#                 if self.fusion_mode == 'add':
#                     fused = patch_latent + dino_latent[:, :, None, None, :]
#                 else:
#                     scale = self.film_scale(dino_latent)
#                     bias = self.film_bias(dino_latent)
#                     fused = patch_latent * (1.0 + scale[:, :, None, None, :]) + bias[:, :, None, None, :]

#                 d_all = self.feature_net(fused)

#             is_base_band = (bb == no_bands - 1)
#             if is_base_band:
#                 d_all *= self.baseband_weight

#             if is_image:
#                 d_all *= self.image_int

#             q_jod -= d_all.view(-1).mean() / no_bands

#         assert not q_jod.isnan()
#         return q_jod


# register_metric(cvvdp_ml_dino)


class cvvdp_ml_dino_fusion(cvvdp_ml_dino_base):
    """
    DINO-fusion model that combines spatial/band features with DINO embeddings.
    
    Architecture:
    1. Embed 8 input features to 128 dimensions using feature_net
    2. Average or max pool embeddings across all spatial patches and bands
    3. Concatenate pooled embeddings (128) with DINO features (768) = 896 dims
    4. Pass through final MLP to predict quality score
    5. Average across frames
    """

    def __init__(self, device=None, pool_type='avg', **kwargs):
        """
        Args:
            device: torch device
            pool_type: 'avg' for average pooling or 'max' for max pooling
        """
        self.set_device(device)
        
        # Feature embedding: 8 features -> 128 dimensions
        embedding_dim = 128
        self.feature_net = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        ).to(self.device)
        
        # Final MLP: (embedding_dim + dino_dim) -> 1
        # embedding_dim=128, dino_dim=768, so input is 896
        dino_dim = 768
        mlp_in_channels = embedding_dim + dino_dim
        self.quality_net = nn.Sequential(
            nn.LayerNorm(mlp_in_channels),
            MLP(
            in_channels=mlp_in_channels,
            hidden_channels=[512, 256, 1],
            activation_layer=torch.nn.ReLU,
            dropout=0.2,
        )).to(self.device)
        
        self.pool_type = pool_type
        self.embedding_dim = embedding_dim
        self.supports_heatmap = True
        
        super().__init__(device=device, **kwargs)
        
        self.train(False)

    def get_nets_to_load(self):
        return ['feature_net', 'quality_net'] if hasattr(self, 'feature_net') else []

    def _clone_video_source(self, vid_source):
        if not hasattr(vid_source, 'test_fname') or not hasattr(vid_source, 'reference_fname'):
            return None

        ctor_kwargs = {
            'test_fname': vid_source.test_fname,
            'reference_fname': vid_source.reference_fname,
            'display_photometry': self.display_photometry,
            'config_paths': getattr(self, 'config_paths', []),
            'fps': getattr(vid_source, 'fps', None),
            'frames': getattr(vid_source, 'in_frames', -1),
            'full_screen_resize': getattr(vid_source, 'full_screen_resize', None),
            'resize_resolution': getattr(vid_source, 'resize_resolution', None),
            'ffmpeg_cc': getattr(vid_source, 'ffmpeg_cc', False),
            'verbose': getattr(vid_source, 'verbose', False),
        }

        try:
            return vid_source.__class__(**ctor_kwargs)
        except TypeError:
            try:
                return vid_source.__class__(
                    vid_source.test_fname,
                    vid_source.reference_fname,
                    display_photometry=self.display_photometry,
                    config_paths=getattr(self, 'config_paths', []),
                    fps=getattr(vid_source, 'fps', None),
                    frames=getattr(vid_source, 'in_frames', -1),
                    full_screen_resize=getattr(vid_source, 'full_screen_resize', None),
                    resize_resolution=getattr(vid_source, 'resize_resolution', None),
                    ffmpeg_cc=getattr(vid_source, 'ffmpeg_cc', False),
                    verbose=getattr(vid_source, 'verbose', False),
                )
            except Exception:
                return None

    def _extract_reference_context(self, vid_source, met_colorspace):
        _, _, no_frames = vid_source.get_video_size()

        ref_frames = []
        for ff in range(no_frames):
            ref_frame = vid_source.get_reference_frame(ff, device=self.device, colorspace=met_colorspace).squeeze(2)
            ref_frames.append(ref_frame[:, 0, :, :])

        return torch.stack(ref_frames, dim=1)

    def _pool_temporal_delta(self, delta, batch, frames):
        delta = delta.reshape(batch, frames)
        return delta.mean(dim=-1).mean()

    def predict_video_source(self, vid_source):
        assert vid_source.get_batch_size()==1 or self.heatmap is None or self.heatmap=='none', 'Heatmaps not supported when batches are used'

        use_ml_heatmap = self.do_heatmap

        dino_vid_source = self._clone_video_source(vid_source)
        context_vid_source = self._clone_video_source(vid_source) if (use_ml_heatmap and self.heatmap != 'raw') else None

        prev_do_heatmap = self.do_heatmap
        if use_ml_heatmap:
            self.do_heatmap = False

        try:
            features, _ = self.extract_features(vid_source)
        finally:
            self.do_heatmap = prev_do_heatmap

        dino_source = dino_vid_source if dino_vid_source is not None else vid_source
        dino_features = self.extract_features_dino(dino_source)

        if use_ml_heatmap:
            q_jod, heatmap_raw = self.do_pooling_and_jods(features, dino_features, return_heatmap=True)
            if self.heatmap == 'raw':
                heatmap = heatmap_raw.detach().type(torch.float16).cpu().unsqueeze(1)
            else:
                met_colorspace = 'logLMS_DKLd65' if self.contrast == 'log' else 'DKLd65'
                context_source = context_vid_source if context_vid_source is not None else vid_source
                ref_context = self._extract_reference_context(context_source, met_colorspace).detach().cpu()
                heatmap = visualize_diff_map(
                    heatmap_raw.detach().cpu(),
                    context_image=ref_context,
                    colormap_type=self.heatmap,
                    use_cpu=False
                ).detach().type(torch.float16).cpu().unsqueeze(0)
        else:
            q_jod = self.do_pooling_and_jods(features, dino_features)
            heatmap = None

        vid_sz = vid_source.get_video_size()
        height, width, n_frames = vid_sz

        stats = {}
        stats['rho_band'] = self.lpyr.get_freqs()
        stats['frames_per_second'] = vid_source.get_frames_per_second()
        stats['width'] = width
        stats['height'] = height
        stats['N_frames'] = n_frames

        if self.dump_channels:
            self.dump_channels.close()

        if use_ml_heatmap:
            stats['heatmap'] = heatmap

        return (q_jod.squeeze(), stats)

    def do_pooling_and_jods(self, features, dino_features, return_heatmap=False):
        """
        Args:
            features: list of tensors, one per band [batch, frames, width, height, channels, stats]
            dino_features: list of tensors [batch, frames, dino_dim] or [batch*frames, dino_dim]
            return_heatmap: if True, return heatmap in addition to quality score
        
        Returns:
            Q_jod: quality score (scalar)
            heatmap_raw (optional): raw heatmap tensor if return_heatmap=True
        """
        no_bands = len(features)
        q_jod = torch.as_tensor(10., device=self.device)

        f0 = features[0]
        ch_dim = 4 if f0.dim() == 6 else 3
        is_image = (f0.shape[ch_dim] == 3)

        dino_ref = dino_features[0]
        if dino_ref.dim() == 2:
            dino_ref = dino_ref.unsqueeze(1)  # [batch*frames, dino_dim] -> [batch, frames, dino_dim] if needed
        dino_ref = dino_ref.to(self.device)

        # Collect pooled embeddings per band to avoid shape mismatch
        # across pyramid bands with different spatial resolutions.
        band_embeddings = []
        heatmap_band_maps = [None] * no_bands if return_heatmap else None
        
        for bb in range(no_bands):
            f = features[bb]

            if is_image:
                if f.dim() == 6:
                    f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], f.shape[3], 1, f.shape[5]), device=self.device)), dim=4)
                else:
                    f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3)

            if self.disabled_features is not None:
                if f.dim() == 6:
                    f[..., self.disabled_features] = 0
                else:
                    f[:, :, :, :, self.disabled_features] = 0

            # Extract and normalize the difference stats
            f_d = f[..., 4:]
            f_d[..., 1] = torch.sqrt(torch.abs(f_d[..., 1]))
            
            # Flatten the last 2 dimensions to get stats concatenated
            f_d = f_d.flatten(start_dim=f_d.dim() - 2)  # flatten the stats dimension
            
            # f_d shape: [batch, frames, width, height, 8] or [batch, width, height, 8]
            if f_d.dim() == 5:
                batch, frames, height, width, _ = f_d.shape
                f_d_flat = f_d.reshape(-1, 8)
                embeddings_tokens = self.feature_net(f_d_flat).reshape(batch, frames, height * width, self.embedding_dim)

                if return_heatmap:
                    dino_cur = dino_ref
                    if dino_cur.dim() == 2:
                        dino_cur = dino_cur.unsqueeze(0)
                    if dino_cur.shape[0] == 1 and batch > 1:
                        dino_cur = dino_cur.expand(batch, -1, -1)
                    if dino_cur.shape[1] == 1 and frames > 1:
                        dino_cur = dino_cur.expand(-1, frames, -1)
                    elif dino_cur.shape[1] > frames:
                        dino_cur = dino_cur[:, :frames, :]
                    elif dino_cur.shape[1] < frames:
                        dino_cur = torch.cat((dino_cur, dino_cur[:, -1:, :].expand(-1, frames - dino_cur.shape[1], -1)), dim=1)

                    dino_map = dino_cur[:, :, None, :].expand(-1, -1, height * width, -1)
                    combined_tokens = torch.cat((embeddings_tokens, dino_map), dim=-1)
                    token_delta = self.quality_net(combined_tokens.reshape(-1, self.embedding_dim + 768)).reshape(batch, frames, height, width)

                    is_base_band = (bb == no_bands - 1)
                    if is_base_band:
                        token_delta *= self.baseband_weight
                    if is_image:
                        token_delta *= self.image_int

                    heatmap_band_maps[bb] = token_delta.clamp_min(0.)

                embeddings = embeddings_tokens

                if self.pool_type == 'avg':
                    embeddings = embeddings.mean(dim=2)
                elif self.pool_type == 'max':
                    embeddings, _ = embeddings.max(dim=2)
                else:
                    raise ValueError(f"Unsupported pooling type: {self.pool_type}")

            elif f_d.dim() == 4:
                batch, height, width, _ = f_d.shape
                frames = 1
                f_d_flat = f_d.reshape(-1, 8)
                embeddings = self.feature_net(f_d_flat).reshape(batch, height * width, self.embedding_dim)

                if self.pool_type == 'avg':
                    embeddings = embeddings.mean(dim=1, keepdim=True)
                elif self.pool_type == 'max':
                    embeddings, _ = embeddings.max(dim=1, keepdim=True)
                else:
                    raise ValueError(f"Unsupported pooling type: {self.pool_type}")

                if return_heatmap:
                    dino_cur = dino_ref
                    if dino_cur.dim() == 2:
                        dino_cur = dino_cur.unsqueeze(0)
                    if dino_cur.shape[0] == 1 and batch > 1:
                        dino_cur = dino_cur.expand(batch, -1, -1)
                    if dino_cur.shape[1] == 1:
                        dino_cur = dino_cur.expand(-1, frames, -1)

                    dino_map = dino_cur[:, :, None, :].expand(-1, -1, height * width, -1)
                    tokens = self.feature_net(f_d_flat).reshape(batch, frames, height * width, self.embedding_dim)
                    combined_tokens = torch.cat((tokens, dino_map), dim=-1)
                    token_delta = self.quality_net(combined_tokens.reshape(-1, self.embedding_dim + 768)).reshape(batch, frames, height, width)

                    is_base_band = (bb == no_bands - 1)
                    if is_base_band:
                        token_delta *= self.baseband_weight
                    if is_image:
                        token_delta *= self.image_int

                    heatmap_band_maps[bb] = token_delta.clamp_min(0.)
            else:
                raise RuntimeError(f"Unsupported feature tensor dimensionality: {f_d.dim()}")

            band_embeddings.append(embeddings)

        # band_embeddings: list of [batch, frames, embedding_dim]
        band_embeddings = torch.stack(band_embeddings, dim=2)  # [batch, frames, bands, embedding_dim]

        baseband_weight = self.baseband_weight
        if isinstance(baseband_weight, torch.Tensor):
            if baseband_weight.numel() > 1:
                baseband_weight = baseband_weight.mean()
            baseband_weight = baseband_weight.to(self.device)

        band_weights = torch.ones((1, 1, no_bands, 1), device=self.device, dtype=band_embeddings.dtype)
        band_weights[:, :, no_bands - 1, :] = baseband_weight
        weighted_band_embeddings = band_embeddings * band_weights

        # Aggregate across bands
        if self.pool_type == 'avg':
            pooled_embeddings = weighted_band_embeddings.sum(dim=2) / band_weights.sum(dim=2).clamp_min(1e-8)
        elif self.pool_type == 'max':
            pooled_embeddings, _ = weighted_band_embeddings.max(dim=2)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool_type}")

        batch, frames, _ = pooled_embeddings.shape

        # Prepare DINO features for concatenation
        if dino_ref.dim() == 2:
            if dino_ref.shape[0] == frames:
                dino_ref = dino_ref.unsqueeze(0)
            else:
                dino_ref = dino_ref.unsqueeze(1)

        if dino_ref.dim() == 3 and dino_ref.shape[0] == 1 and batch > 1:
            dino_ref = dino_ref.expand(batch, -1, -1)

        # Ensure dimensional consistency
        if dino_ref.shape[1] != frames:
            if dino_ref.shape[1] == 1:
                dino_ref = dino_ref.expand(-1, frames, -1)
            else:
                min_frames = min(dino_ref.shape[1], frames)
                dino_ref = dino_ref[:, :min_frames, :]
                pooled_embeddings = pooled_embeddings[:, :min_frames, :]
                frames = min_frames

        # Concatenate pooled embeddings with DINO features
        combined_features = torch.cat((pooled_embeddings, dino_ref), dim=-1)  # [batch, frames, 896]
        
        # Reshape for MLP
        combined_flat = combined_features.reshape(-1, self.embedding_dim + 768)
        
        # Predict quality differences
        delta = self.quality_net(combined_flat)  # [batch*frames, 1]

        # Reshape back and average across frames
        delta = self._pool_temporal_delta(delta, batch, frames)

        if is_image:
            delta *= self.image_int

        q_jod = q_jod - delta

        assert not q_jod.isnan()
        if not return_heatmap:
            return q_jod

        heatmap_pyr = lpyr_dec_2(self.lpyr.W, self.lpyr.H, self.pix_per_deg, self.device)
        heatmap_pyr.decompose(torch.zeros((1, 1, self.lpyr.H, self.lpyr.W), dtype=features[0].dtype, device=self.device))

        for bb in range(no_bands):
            band_map = heatmap_band_maps[bb]
            target_h = heatmap_pyr.get_lband(bb).shape[-2]
            target_w = heatmap_pyr.get_lband(bb).shape[-1]

            if band_map.shape[-2] != target_h or band_map.shape[-1] != target_w:
                bsz, nframes, src_h, src_w = band_map.shape
                band_map = Func.adaptive_avg_pool2d(
                    band_map.reshape(bsz * nframes, 1, src_h, src_w),
                    output_size=(target_h, target_w)
                ).reshape(bsz, nframes, target_h, target_w)

            heatmap_band_maps[bb] = band_map

        n_frames = heatmap_band_maps[0].shape[1]
        heatmap_raw = torch.empty((batch, n_frames, self.lpyr.H, self.lpyr.W), dtype=features[0].dtype, device=self.device)

        with torch.no_grad():
            for bidx in range(batch):
                for ff in range(n_frames):
                    frame_pyr = lpyr_dec_2(self.lpyr.W, self.lpyr.H, self.pix_per_deg, self.device)
                    frame_pyr.decompose(torch.zeros((1, 1, self.lpyr.H, self.lpyr.W), dtype=features[0].dtype, device=self.device))
                    for bb in range(no_bands):
                        frame_pyr.set_lband(bb, heatmap_band_maps[bb][bidx:bidx+1, ff:ff+1, :, :])
                    frame_map = 1. - (self.met2jod(frame_pyr.reconstruct()) / 10.)
                    heatmap_raw[bidx:bidx+1, ff:ff+1, :, :] = frame_map

        return q_jod, heatmap_raw


register_metric(cvvdp_ml_dino_fusion)


class cvvdp_ml_dino_fusion_temporal_weighted(cvvdp_ml_dino_fusion):
    """
    Plain DINO-fusion model with weighted temporal pooling.

    The temporal weights follow the same idea as ColorVideoVDP weighted pooling:
        w = x / (x + x0), with x0 = 10**wp_base,
    and normalized averaging with epsilon = 10**wp_epsilon.
    """

    def __init__(self, device=None, pool_type='avg', wp_base=-1.0, wp_epsilon=-5.0, **kwargs):
        self.wp_base = torch.as_tensor(wp_base, dtype=torch.float32)
        self.wp_epsilon = torch.as_tensor(wp_epsilon, dtype=torch.float32)
        super().__init__(device=device, pool_type=pool_type, **kwargs)

    def _pool_temporal_delta(self, delta, batch, frames):
        delta = delta.reshape(batch, frames)
        if frames <= 1:
            return delta.mean(dim=-1).mean()

        wp_base = self.wp_base.to(device=delta.device, dtype=delta.dtype)
        wp_epsilon = self.wp_epsilon.to(device=delta.device, dtype=delta.dtype)

        x = delta.abs()
        x_0 = torch.pow(torch.as_tensor(10.0, device=delta.device, dtype=delta.dtype), wp_base)
        w = x / (x + x_0)

        eps = torch.pow(torch.as_tensor(10.0, device=delta.device, dtype=delta.dtype), wp_epsilon)
        weighted_delta = (delta * w).sum(dim=-1) / (w.sum(dim=-1) + eps)
        return weighted_delta.mean()


register_metric(cvvdp_ml_dino_fusion_temporal_weighted)


class cvvdp_ml_dino_fusion_temporal_multiscale_pooling(cvvdp_ml_dino_fusion):
    """
    Plain DINO-fusion model with multiscale temporal pooling.

        Temporal pooling strategy:
        - Build overlapping segments for windows [0.1s, 0.5s, 1.0s]
            (50% overlap by default)
    - For each window scale, compute per-segment statistics (mean and p20)
        - Select a soft-worst segment score for each statistic (minimum quality emphasized)
    - Average across all categories and across the batch
    """

    def __init__(self,
                 device=None,
                 pool_type='avg',
                 temporal_windows_s=(0.05, 1.0),
                 temporal_percentile=0.2,
                 temporal_overlap=0.5,
                 soft_worst_temperature=10.0,
                 temporal_use_mean=True,
                 temporal_use_percentile=False,
                 temporal_fps_hint=None,
                 **kwargs):
        self.temporal_windows_s = tuple(float(w) for w in temporal_windows_s)
        self.temporal_percentile = float(temporal_percentile)
        self.temporal_overlap = float(temporal_overlap)
        self.soft_worst_temperature = float(soft_worst_temperature)
        self.temporal_use_mean = bool(temporal_use_mean)
        self.temporal_use_percentile = bool(temporal_use_percentile)
        self.temporal_fps_hint = temporal_fps_hint
        self._temporal_pool_fps = None
        super().__init__(device=device, pool_type=pool_type, **kwargs)

    def predict_video_source(self, vid_source):
        try:
            fps = float(vid_source.get_frames_per_second())
            if not math.isfinite(fps) or fps <= 0:
                fps = None
        except Exception:
            fps = None

        self._temporal_pool_fps = fps
        try:
            return super().predict_video_source(vid_source)
        finally:
            self._temporal_pool_fps = None

    def _soft_worst(self, values, dim=-1, mode='max'):
        if values.shape[dim] <= 1:
            return values.squeeze(dim)

        tau = torch.as_tensor(self.soft_worst_temperature, device=values.device, dtype=values.dtype)
        if not torch.isfinite(tau) or tau <= 0:
            return values.max(dim=dim).values if mode == 'max' else values.min(dim=dim).values

        if mode == 'max':
            ref_vals = values.max(dim=dim, keepdim=True).values
            logits = (values - ref_vals) * tau
        elif mode == 'min':
            ref_vals = values.min(dim=dim, keepdim=True).values
            logits = -(values - ref_vals) * tau
        else:
            raise ValueError(f"Unsupported soft-worst mode: {mode}")

        weights = torch.softmax(logits, dim=dim)
        return (weights * values).sum(dim=dim)

    def _resolve_temporal_pool_fps(self):
        candidates = [
            self._temporal_pool_fps,
            self.temporal_fps_hint,
            getattr(self, 'fps', None),
            getattr(self, 'frames_per_second', None),
            getattr(self, 'source_fps', None),
            getattr(self, 'nominal_fps', None),
        ]

        for cand in candidates:
            try:
                fps = float(cand)
                if math.isfinite(fps) and fps > 0:
                    return fps
            except Exception:
                continue

        return 30.0

    def _pool_temporal_delta(self, delta, batch, frames):
        delta = delta.reshape(batch, frames)
        if frames <= 1:
            return delta.mean(dim=-1).mean()

        use_mean = self.temporal_use_mean
        use_percentile = self.temporal_use_percentile
        if not use_mean and not use_percentile:
            use_mean = True

        p = max(0.0, min(1.0, self.temporal_percentile))
        overlap = max(0.0, min(0.95, self.temporal_overlap))
        fps = self._resolve_temporal_pool_fps()

        category_values = []

        for window_s in self.temporal_windows_s:
            window_frames = max(1, int(round(window_s * fps)))
            hop_frames = max(1, int(round(window_frames * (1.0 - overlap))))

            seg_mean_vals = []
            seg_perc_vals = []

            for start in range(0, frames, hop_frames):
                end = min(start + window_frames, frames)
                seg = delta[:, start:end]
                if seg.shape[1] == 0:
                    continue

                if use_mean:
                    seg_mean_vals.append(seg.mean(dim=-1))

                if use_percentile:
                    seg_perc_vals.append(torch.quantile(seg, q=p, dim=-1))

            if use_mean and len(seg_mean_vals) > 0:
                seg_mean = torch.stack(seg_mean_vals, dim=-1)
                seg_mean_quality = 10.0 - seg_mean
                worst_mean_quality = self._soft_worst(seg_mean_quality, dim=-1, mode='min')
                worst_mean_delta = 10.0 - worst_mean_quality
                category_values.append(worst_mean_delta)

            if use_percentile and len(seg_perc_vals) > 0:
                seg_perc = torch.stack(seg_perc_vals, dim=-1)
                seg_perc_quality = 10.0 - seg_perc
                worst_perc_quality = self._soft_worst(seg_perc_quality, dim=-1, mode='min')
                worst_perc_delta = 10.0 - worst_perc_quality
                category_values.append(worst_perc_delta)

        if len(category_values) == 0:
            return delta.mean(dim=-1).mean()

        pooled = torch.stack(category_values, dim=-1).mean(dim=-1)
        return pooled.mean()


register_metric(cvvdp_ml_dino_fusion_temporal_multiscale_pooling)


class cvvdp_ml_dino_fusion_temporal_hierarchical_topk(cvvdp_ml_dino_fusion):
    """
    DINO-fusion model with hierarchical temporal pooling.

    Pooling strategy:
    - Split the sequence into coarse windows (e.g., 1.0s)
    - For each coarse window, compute its mean delta
    - Inside each coarse window, compute fine-window means (e.g., 0.1s)
    - Take top-k worst fine-window means (largest deltas / lowest quality)
    - Fuse coarse mean with worst fine-window statistic
    - Average across coarse windows and across batch
    """

    def __init__(self,
                 device=None,
                 pool_type='avg',
                 coarse_window_s=1.0,
                 coarse_overlap=0.5,
                 fine_window_s=0.1,
                 fine_overlap=0.5,
                 topk_worst=1,
                 topk_mode='mean',
                 topk_softmax_temperature=10.0,
                 coarse_fine_fusion_alpha=0.5,
                 temporal_fps_hint=None,
                 **kwargs):
        self.coarse_window_s = float(coarse_window_s)
        self.coarse_overlap = float(coarse_overlap)
        self.fine_window_s = float(fine_window_s)
        self.fine_overlap = float(fine_overlap)
        self.topk_worst = int(topk_worst)
        self.topk_mode = str(topk_mode)
        self.topk_softmax_temperature = float(topk_softmax_temperature)
        self.coarse_fine_fusion_alpha = float(coarse_fine_fusion_alpha)
        self.temporal_fps_hint = temporal_fps_hint
        self._temporal_pool_fps = None
        super().__init__(device=device, pool_type=pool_type, **kwargs)

    def predict_video_source(self, vid_source):
        try:
            fps = float(vid_source.get_frames_per_second())
            if not math.isfinite(fps) or fps <= 0:
                fps = None
        except Exception:
            fps = None

        self._temporal_pool_fps = fps
        try:
            return super().predict_video_source(vid_source)
        finally:
            self._temporal_pool_fps = None

    def _resolve_temporal_pool_fps(self):
        candidates = [
            self._temporal_pool_fps,
            self.temporal_fps_hint,
            getattr(self, 'fps', None),
            getattr(self, 'frames_per_second', None),
            getattr(self, 'source_fps', None),
            getattr(self, 'nominal_fps', None),
        ]

        for cand in candidates:
            try:
                fps = float(cand)
                if math.isfinite(fps) and fps > 0:
                    return fps
            except Exception:
                continue

        return 30.0

    def _reduce_worst_fine_values(self, fine_values):
        # fine_values: [batch, k], higher values are worse (larger deltas)
        if fine_values.shape[-1] <= 1:
            return fine_values.squeeze(-1)

        if self.topk_mode == 'max':
            return fine_values.max(dim=-1).values

        if self.topk_mode == 'mean':
            return fine_values.mean(dim=-1)

        if self.topk_mode == 'softmax':
            tau = torch.as_tensor(self.topk_softmax_temperature, device=fine_values.device, dtype=fine_values.dtype)
            if not torch.isfinite(tau) or tau <= 0:
                return fine_values.mean(dim=-1)

            ref_vals = fine_values.max(dim=-1, keepdim=True).values
            logits = (fine_values - ref_vals) * tau
            weights = torch.softmax(logits, dim=-1)
            return (weights * fine_values).sum(dim=-1)

        raise ValueError(f"Unsupported topk_mode: {self.topk_mode}. Expected one of ['max', 'mean', 'softmax']")

    def _pool_temporal_delta(self, delta, batch, frames):
        delta = delta.reshape(batch, frames)
        if frames <= 1:
            return delta.mean(dim=-1).mean()

        fps = self._resolve_temporal_pool_fps()
        coarse_window_frames = max(1, int(round(self.coarse_window_s * fps)))
        fine_window_frames = max(1, int(round(self.fine_window_s * fps)))

        coarse_overlap = max(0.0, min(0.95, self.coarse_overlap))
        fine_overlap = max(0.0, min(0.95, self.fine_overlap))
        coarse_hop = max(1, int(round(coarse_window_frames * (1.0 - coarse_overlap))))
        fine_hop = max(1, int(round(fine_window_frames * (1.0 - fine_overlap))))

        k_worst = max(1, self.topk_worst)
        fusion_alpha = max(0.0, min(1.0, self.coarse_fine_fusion_alpha))
        fusion_alpha = torch.as_tensor(fusion_alpha, dtype=delta.dtype, device=delta.device)

        coarse_scores = []

        for coarse_start in range(0, frames, coarse_hop):
            coarse_end = min(coarse_start + coarse_window_frames, frames)
            coarse_seg = delta[:, coarse_start:coarse_end]
            if coarse_seg.shape[1] == 0:
                continue

            coarse_mean = coarse_seg.mean(dim=-1)

            fine_seg_means = []
            for fine_start in range(coarse_start, coarse_end, fine_hop):
                fine_end = min(fine_start + fine_window_frames, coarse_end)
                fine_seg = delta[:, fine_start:fine_end]
                if fine_seg.shape[1] == 0:
                    continue
                fine_seg_means.append(fine_seg.mean(dim=-1))

            if len(fine_seg_means) == 0:
                adjusted_coarse = coarse_mean
            else:
                fine_seg_means = torch.stack(fine_seg_means, dim=-1)
                k = min(k_worst, fine_seg_means.shape[-1])
                worst_k = torch.topk(fine_seg_means, k=k, dim=-1, largest=True).values
                worst_summary = self._reduce_worst_fine_values(worst_k)
                adjusted_coarse = (1.0 - fusion_alpha) * coarse_mean + fusion_alpha * worst_summary

            coarse_scores.append(adjusted_coarse)

        if len(coarse_scores) == 0:
            return delta.mean(dim=-1).mean()

        return torch.stack(coarse_scores, dim=-1).mean(dim=-1).mean()


register_metric(cvvdp_ml_dino_fusion_temporal_hierarchical_topk)


class cvvdp_ml_dino_fusion_bandfreq(cvvdp_ml_dino_fusion):
    """
    DINO-fusion model that injects the spatial band frequency into the patch features.

    Compared with `cvvdp_ml_dino_fusion`, this variant does not use a baseband weight.
    Instead, each band's scalar spatial frequency is appended to the D features before
    the feature embedding MLP.
    """

    def __init__(self, device=None, pool_type='avg', band_freqs=None, **kwargs):
        self.set_device(device)

        embedding_dim = 128
        self.feature_input_dim = 9  # 8 D features + 1 band-frequency scalar
        self.feature_net = nn.Sequential(
            nn.Linear(self.feature_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        ).to(self.device)

        dino_dim = 768
        mlp_in_channels = embedding_dim + dino_dim
        self.quality_net = nn.Sequential(
            nn.LayerNorm(mlp_in_channels),
            MLP(
                in_channels=mlp_in_channels,
                hidden_channels=[512, 256, 1],
                activation_layer=torch.nn.ReLU,
                dropout=0.2,
            ),
        ).to(self.device)

        self.pool_type = pool_type
        self.embedding_dim = embedding_dim
        self.supports_heatmap = True
        self.band_freqs = band_freqs

        cvvdp_ml_dino_base.__init__(self, device=device, **kwargs)

        self.train(False)

    def _append_band_frequency(self, f_d, band_freq):
        band_freq = torch.as_tensor(band_freq, device=self.device, dtype=f_d.dtype)
        freq_map = torch.ones(f_d.shape[:-1] + (1,), device=self.device, dtype=f_d.dtype) * band_freq
        return torch.cat((f_d, freq_map), dim=-1)

    def _resolve_band_freqs(self, no_bands):
        if self.band_freqs is not None:
            band_freqs = torch.as_tensor(self.band_freqs, device=self.device, dtype=torch.float32).flatten()
            if band_freqs.numel() == 1:
                band_freqs = band_freqs.expand(no_bands)
            elif band_freqs.numel() < no_bands:
                tail = band_freqs[-1:].expand(no_bands - band_freqs.numel())
                band_freqs = torch.cat((band_freqs, tail), dim=0)
            else:
                band_freqs = band_freqs[:no_bands]
            return band_freqs

        # Match ColorVideoVDP's pyramid frequency definition.
        # lpyr_dec uses [1.0] + [0.3228 * 2**(-f) for f in range(height)] times ppd/2.
        # We prioritize current-sample pix_per_deg so feature-mode pooling is not affected
        # by a stale pyramid object from earlier feature extraction.
        ppd = getattr(self, 'pix_per_deg', None)
        if ppd is not None:
            ppd = float(ppd)
        if ppd is not None and np.isfinite(ppd) and ppd > 0:
            height = max(int(no_bands) - 1, 0)
            band_freqs = np.array([1.0] + [0.3228 * 2.0 ** (-f) for f in range(height)], dtype=np.float32) * (ppd / 2.0)
            if band_freqs.shape[0] < no_bands:
                tail = np.repeat(band_freqs[-1:], no_bands - band_freqs.shape[0])
                band_freqs = np.concatenate([band_freqs, tail], axis=0)
            return torch.as_tensor(band_freqs[:no_bands], device=self.device)

        if getattr(self, 'lpyr', None) is not None:
            band_freqs = torch.as_tensor(self.lpyr.get_freqs(), device=self.device, dtype=torch.float32).flatten()
            if band_freqs.numel() > 0:
                if band_freqs.numel() >= no_bands:
                    return band_freqs[:no_bands]
                tail = band_freqs[-1:].expand(no_bands - band_freqs.numel())
                return torch.cat((band_freqs, tail), dim=0)

        ppd = 1.0
        height = max(int(no_bands) - 1, 0)
        band_freqs = np.array([1.0] + [0.3228 * 2.0 ** (-f) for f in range(height)], dtype=np.float32) * (ppd / 2.0)
        if band_freqs.shape[0] < no_bands:
            tail = np.repeat(band_freqs[-1:], no_bands - band_freqs.shape[0])
            band_freqs = np.concatenate([band_freqs, tail], axis=0)
        return torch.as_tensor(band_freqs[:no_bands], device=self.device)

    def _pool_temporal_delta(self, delta, batch, frames):
        delta = delta.reshape(batch, frames)
        return delta.mean(dim=-1).mean()

    def do_pooling_and_jods(self, features, dino_features, return_heatmap=False):
        no_bands = len(features)
        q_jod = torch.as_tensor(10., device=self.device)

        f0 = features[0]
        ch_dim = 4 if f0.dim() == 6 else 3
        is_image = (f0.shape[ch_dim] == 3)

        dino_ref = dino_features[0]
        if dino_ref.dim() == 2:
            dino_ref = dino_ref.unsqueeze(1)
        dino_ref = dino_ref.to(self.device)

        rho_band = self._resolve_band_freqs(no_bands)

        band_embeddings = []
        heatmap_band_maps = [None] * no_bands if return_heatmap else None

        for bb in range(no_bands):
            f = features[bb]

            if is_image:
                if f.dim() == 6:
                    f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], f.shape[3], 1, f.shape[5]), device=self.device)), dim=4)
                else:
                    f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3)

            if self.disabled_features is not None:
                if f.dim() == 6:
                    f[..., self.disabled_features] = 0
                else:
                    f[:, :, :, :, self.disabled_features] = 0

            # Extract the D-related statistics and append the band frequency as an extra feature.
            f_d = f[..., 4:]
            f_d[..., 1] = torch.sqrt(torch.abs(f_d[..., 1]))
            f_d = f_d.flatten(start_dim=f_d.dim() - 2)
            f_d = self._append_band_frequency(f_d, rho_band[bb])

            if f_d.dim() == 5:
                batch, frames, height, width, feat_dim = f_d.shape
                f_d_flat = f_d.reshape(-1, feat_dim)
                embeddings_tokens = self.feature_net(f_d_flat).reshape(batch, frames, height * width, self.embedding_dim)

                if return_heatmap:
                    dino_cur = dino_ref
                    if dino_cur.dim() == 2:
                        dino_cur = dino_cur.unsqueeze(0)
                    if dino_cur.shape[0] == 1 and batch > 1:
                        dino_cur = dino_cur.expand(batch, -1, -1)
                    if dino_cur.shape[1] == 1 and frames > 1:
                        dino_cur = dino_cur.expand(-1, frames, -1)
                    elif dino_cur.shape[1] > frames:
                        dino_cur = dino_cur[:, :frames, :]
                    elif dino_cur.shape[1] < frames:
                        dino_cur = torch.cat((dino_cur, dino_cur[:, -1:, :].expand(-1, frames - dino_cur.shape[1], -1)), dim=1)

                    dino_map = dino_cur[:, :, None, :].expand(-1, -1, height * width, -1)
                    combined_tokens = torch.cat((embeddings_tokens, dino_map), dim=-1)
                    token_delta = self.quality_net(combined_tokens.reshape(-1, self.embedding_dim + 768)).reshape(batch, frames, height, width)

                    if is_image:
                        token_delta *= self.image_int

                    heatmap_band_maps[bb] = token_delta.clamp_min(0.)

                embeddings = embeddings_tokens

                if self.pool_type == 'avg':
                    embeddings = embeddings.mean(dim=2)
                elif self.pool_type == 'max':
                    embeddings, _ = embeddings.max(dim=2)
                else:
                    raise ValueError(f"Unsupported pooling type: {self.pool_type}")

            elif f_d.dim() == 4:
                batch, height, width, feat_dim = f_d.shape
                frames = 1
                f_d_flat = f_d.reshape(-1, feat_dim)
                embeddings_tokens = self.feature_net(f_d_flat).reshape(batch, frames, height * width, self.embedding_dim)

                if self.pool_type == 'avg':
                    embeddings = embeddings_tokens.mean(dim=2).squeeze(1)
                    embeddings = embeddings.unsqueeze(1)
                elif self.pool_type == 'max':
                    embeddings, _ = embeddings_tokens.max(dim=2)
                else:
                    raise ValueError(f"Unsupported pooling type: {self.pool_type}")

                if return_heatmap:
                    dino_cur = dino_ref
                    if dino_cur.dim() == 2:
                        dino_cur = dino_cur.unsqueeze(0)
                    if dino_cur.shape[0] == 1 and batch > 1:
                        dino_cur = dino_cur.expand(batch, -1, -1)
                    if dino_cur.shape[1] == 1:
                        dino_cur = dino_cur.expand(-1, frames, -1)

                    dino_map = dino_cur[:, :, None, :].expand(-1, -1, height * width, -1)
                    combined_tokens = torch.cat((embeddings_tokens, dino_map), dim=-1)
                    token_delta = self.quality_net(combined_tokens.reshape(-1, self.embedding_dim + 768)).reshape(batch, frames, height, width)

                    if is_image:
                        token_delta *= self.image_int

                    heatmap_band_maps[bb] = token_delta.clamp_min(0.)
            else:
                raise RuntimeError(f"Unsupported feature tensor dimensionality: {f_d.dim()}")

            band_embeddings.append(embeddings)

        band_embeddings = torch.stack(band_embeddings, dim=2)

        if self.pool_type == 'avg':
            pooled_embeddings = band_embeddings.mean(dim=2)
        elif self.pool_type == 'max':
            pooled_embeddings, _ = band_embeddings.max(dim=2)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool_type}")

        batch, frames, _ = pooled_embeddings.shape

        if dino_ref.dim() == 2:
            if dino_ref.shape[0] == frames:
                dino_ref = dino_ref.unsqueeze(0)
            else:
                dino_ref = dino_ref.unsqueeze(1)

        if dino_ref.dim() == 3 and dino_ref.shape[0] == 1 and batch > 1:
            dino_ref = dino_ref.expand(batch, -1, -1)

        if dino_ref.shape[1] != frames:
            if dino_ref.shape[1] == 1:
                dino_ref = dino_ref.expand(-1, frames, -1)
            else:
                min_frames = min(dino_ref.shape[1], frames)
                dino_ref = dino_ref[:, :min_frames, :]
                pooled_embeddings = pooled_embeddings[:, :min_frames, :]
                frames = min_frames

        combined_features = torch.cat((pooled_embeddings, dino_ref), dim=-1)
        combined_flat = combined_features.reshape(-1, self.embedding_dim + 768)
        delta = self.quality_net(combined_flat)
        delta = self._pool_temporal_delta(delta, batch, frames)

        if is_image:
            delta *= self.image_int

        q_jod = q_jod - delta

        assert not q_jod.isnan()
        if not return_heatmap:
            return q_jod

        heatmap_pyr = lpyr_dec_2(self.lpyr.W, self.lpyr.H, self.pix_per_deg, self.device)
        heatmap_pyr.decompose(torch.zeros((1, 1, self.lpyr.H, self.lpyr.W), dtype=features[0].dtype, device=self.device))

        for bb in range(no_bands):
            band_map = heatmap_band_maps[bb]
            target_h = heatmap_pyr.get_lband(bb).shape[-2]
            target_w = heatmap_pyr.get_lband(bb).shape[-1]

            if band_map.shape[-2] != target_h or band_map.shape[-1] != target_w:
                bsz, nframes, src_h, src_w = band_map.shape
                band_map = Func.adaptive_avg_pool2d(
                    band_map.reshape(bsz * nframes, 1, src_h, src_w),
                    output_size=(target_h, target_w)
                ).reshape(bsz, nframes, target_h, target_w)

            heatmap_band_maps[bb] = band_map

        n_frames = heatmap_band_maps[0].shape[1]
        heatmap_raw = torch.empty((batch, n_frames, self.lpyr.H, self.lpyr.W), dtype=features[0].dtype, device=self.device)

        with torch.no_grad():
            for bidx in range(batch):
                for ff in range(n_frames):
                    frame_pyr = lpyr_dec_2(self.lpyr.W, self.lpyr.H, self.pix_per_deg, self.device)
                    frame_pyr.decompose(torch.zeros((1, 1, self.lpyr.H, self.lpyr.W), dtype=features[0].dtype, device=self.device))
                    for bb in range(no_bands):
                        frame_pyr.set_lband(bb, heatmap_band_maps[bb][bidx:bidx+1, ff:ff+1, :, :])
                    frame_map = 1. - (self.met2jod(frame_pyr.reconstruct()) / 10.)
                    heatmap_raw[bidx:bidx+1, ff:ff+1, :, :] = frame_map

        return q_jod, heatmap_raw


register_metric(cvvdp_ml_dino_fusion_bandfreq)


class cvvdp_ml_dino_fusion_bandfreq_temporal_weighted(cvvdp_ml_dino_fusion_bandfreq):
    """
    Band-frequency DINO fusion with weighted temporal pooling.

    Uses the same weighting idea as weighted-pooling cvvdp:
        w = x / (x + x0), where x0 = 10**wp_base,
    and normalized weighted averaging with epsilon = 10**wp_epsilon.
    """

    def __init__(self, device=None, pool_type='avg', band_freqs=None, wp_base=-1.0, wp_epsilon=-5.0, **kwargs):
        self.wp_base = torch.as_tensor(wp_base, dtype=torch.float32)
        self.wp_epsilon = torch.as_tensor(wp_epsilon, dtype=torch.float32)
        super().__init__(device=device, pool_type=pool_type, band_freqs=band_freqs, **kwargs)

    def _pool_temporal_delta(self, delta, batch, frames):
        delta = delta.reshape(batch, frames)
        if frames <= 1:
            return delta.mean(dim=-1).mean()

        wp_base = self.wp_base.to(device=delta.device, dtype=delta.dtype)
        wp_epsilon = self.wp_epsilon.to(device=delta.device, dtype=delta.dtype)

        x = delta.abs()
        x_0 = torch.pow(torch.as_tensor(10.0, device=delta.device, dtype=delta.dtype), wp_base)
        w = x / (x + x_0)

        eps = torch.pow(torch.as_tensor(10.0, device=delta.device, dtype=delta.dtype), wp_epsilon)
        weighted_delta = (delta * w).sum(dim=-1) / (w.sum(dim=-1) + eps)
        return weighted_delta.mean()


register_metric(cvvdp_ml_dino_fusion_bandfreq_temporal_weighted)


# class cvvdp_ml_dino_fusion_simplified(cvvdp_ml_dino_base):
#     """
#     Simplified DINO-fusion model with lightweight linear heads.
#     """

#     def __init__(self, device=None, pool_type='avg', **kwargs):
#         self.set_device(device)

#         embedding_dim = 128
#         self.feature_net = nn.Linear(8, embedding_dim).to(self.device)

#         dino_dim = 768
#         mlp_in_channels = embedding_dim + dino_dim
#         self.quality_net = nn.Sequential(
#             nn.LayerNorm(mlp_in_channels),
#             nn.Linear(mlp_in_channels, 1),
#             nn.ReLU(),
#         ).to(self.device)

#         self.pool_type = pool_type
#         self.embedding_dim = embedding_dim
#         self.supports_heatmap = True

#         super().__init__(device=device, **kwargs)

#         self.train(False)

#     def get_nets_to_load(self):
#         return ['feature_net', 'quality_net'] if hasattr(self, 'feature_net') else []

#     def _clone_video_source(self, vid_source):
#         if not hasattr(vid_source, 'test_fname') or not hasattr(vid_source, 'reference_fname'):
#             return None

#         ctor_kwargs = {
#             'test_fname': vid_source.test_fname,
#             'reference_fname': vid_source.reference_fname,
#             'display_photometry': self.display_photometry,
#             'config_paths': getattr(self, 'config_paths', []),
#             'fps': getattr(vid_source, 'fps', None),
#             'frames': getattr(vid_source, 'in_frames', -1),
#             'full_screen_resize': getattr(vid_source, 'full_screen_resize', None),
#             'resize_resolution': getattr(vid_source, 'resize_resolution', None),
#             'ffmpeg_cc': getattr(vid_source, 'ffmpeg_cc', False),
#             'verbose': getattr(vid_source, 'verbose', False),
#         }

#         try:
#             return vid_source.__class__(**ctor_kwargs)
#         except TypeError:
#             try:
#                 return vid_source.__class__(
#                     vid_source.test_fname,
#                     vid_source.reference_fname,
#                     display_photometry=self.display_photometry,
#                     config_paths=getattr(self, 'config_paths', []),
#                     fps=getattr(vid_source, 'fps', None),
#                     frames=getattr(vid_source, 'in_frames', -1),
#                     full_screen_resize=getattr(vid_source, 'full_screen_resize', None),
#                     resize_resolution=getattr(vid_source, 'resize_resolution', None),
#                     ffmpeg_cc=getattr(vid_source, 'ffmpeg_cc', False),
#                     verbose=getattr(vid_source, 'verbose', False),
#                 )
#             except Exception:
#                 return None

#     def _extract_reference_context(self, vid_source, met_colorspace):
#         _, _, no_frames = vid_source.get_video_size()

#         ref_frames = []
#         for ff in range(no_frames):
#             ref_frame = vid_source.get_reference_frame(ff, device=self.device, colorspace=met_colorspace).squeeze(2)
#             ref_frames.append(ref_frame[:, 0, :, :])

#         return torch.stack(ref_frames, dim=1)

#     def predict_video_source(self, vid_source):
#         assert vid_source.get_batch_size()==1 or self.heatmap is None or self.heatmap=='none', 'Heatmaps not supported when batches are used'

#         use_ml_heatmap = self.do_heatmap

#         dino_vid_source = self._clone_video_source(vid_source)
#         context_vid_source = self._clone_video_source(vid_source) if (use_ml_heatmap and self.heatmap != 'raw') else None

#         prev_do_heatmap = self.do_heatmap
#         if use_ml_heatmap:
#             self.do_heatmap = False

#         try:
#             features, _ = self.extract_features(vid_source)
#         finally:
#             self.do_heatmap = prev_do_heatmap

#         dino_source = dino_vid_source if dino_vid_source is not None else vid_source
#         dino_features = self.extract_features_dino(dino_source)

#         if use_ml_heatmap:
#             q_jod, heatmap_raw = self.do_pooling_and_jods(features, dino_features, return_heatmap=True)
#             if self.heatmap == 'raw':
#                 heatmap = heatmap_raw.detach().type(torch.float16).cpu().unsqueeze(1)
#             else:
#                 met_colorspace = 'logLMS_DKLd65' if self.contrast == 'log' else 'DKLd65'
#                 context_source = context_vid_source if context_vid_source is not None else vid_source
#                 ref_context = self._extract_reference_context(context_source, met_colorspace).detach().cpu()
#                 heatmap = visualize_diff_map(
#                     heatmap_raw.detach().cpu(),
#                     context_image=ref_context,
#                     colormap_type=self.heatmap,
#                     use_cpu=False
#                 ).detach().type(torch.float16).cpu().unsqueeze(0)
#         else:
#             q_jod = self.do_pooling_and_jods(features, dino_features)
#             heatmap = None

#         vid_sz = vid_source.get_video_size()
#         height, width, n_frames = vid_sz

#         stats = {}
#         stats['rho_band'] = self.lpyr.get_freqs()
#         stats['frames_per_second'] = vid_source.get_frames_per_second()
#         stats['width'] = width
#         stats['height'] = height
#         stats['N_frames'] = n_frames

#         if self.dump_channels:
#             self.dump_channels.close()

#         if use_ml_heatmap:
#             stats['heatmap'] = heatmap

#         return (q_jod.squeeze(), stats)

#     def do_pooling_and_jods(self, features, dino_features, return_heatmap=False):
#         no_bands = len(features)
#         q_jod = torch.as_tensor(10., device=self.device)

#         f0 = features[0]
#         ch_dim = 4 if f0.dim() == 6 else 3
#         is_image = (f0.shape[ch_dim] == 3)

#         dino_ref = dino_features[0]
#         if dino_ref.dim() == 2:
#             dino_ref = dino_ref.unsqueeze(1)
#         dino_ref = dino_ref.to(self.device)

#         band_embeddings = []
#         heatmap_band_maps = [None] * no_bands if return_heatmap else None

#         for bb in range(no_bands):
#             f = features[bb]

#             if is_image:
#                 if f.dim() == 6:
#                     f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], f.shape[3], 1, f.shape[5]), device=self.device)), dim=4)
#                 else:
#                     f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], 1, f.shape[4]), device=self.device)), dim=3)

#             if self.disabled_features is not None:
#                 if f.dim() == 6:
#                     f[..., self.disabled_features] = 0
#                 else:
#                     f[:, :, :, :, self.disabled_features] = 0

#             f_d = f[..., 4:]
#             f_d[..., 1] = torch.sqrt(torch.abs(f_d[..., 1]))
#             f_d = f_d.flatten(start_dim=f_d.dim() - 2)

#             if f_d.dim() == 5:
#                 batch, frames, height, width, _ = f_d.shape
#                 f_d_flat = f_d.reshape(-1, 8)
#                 embeddings_tokens = self.feature_net(f_d_flat).reshape(batch, frames, height * width, self.embedding_dim)

#                 if return_heatmap:
#                     dino_cur = dino_ref
#                     if dino_cur.dim() == 2:
#                         dino_cur = dino_cur.unsqueeze(0)
#                     if dino_cur.shape[0] == 1 and batch > 1:
#                         dino_cur = dino_cur.expand(batch, -1, -1)
#                     if dino_cur.shape[1] == 1 and frames > 1:
#                         dino_cur = dino_cur.expand(-1, frames, -1)
#                     elif dino_cur.shape[1] > frames:
#                         dino_cur = dino_cur[:, :frames, :]
#                     elif dino_cur.shape[1] < frames:
#                         dino_cur = torch.cat((dino_cur, dino_cur[:, -1:, :].expand(-1, frames - dino_cur.shape[1], -1)), dim=1)

#                     dino_map = dino_cur[:, :, None, :].expand(-1, -1, height * width, -1)
#                     combined_tokens = torch.cat((embeddings_tokens, dino_map), dim=-1)
#                     token_delta = self.quality_net(combined_tokens.reshape(-1, self.embedding_dim + 768)).reshape(batch, frames, height, width)

#                     is_base_band = (bb == no_bands - 1)
#                     if is_base_band:
#                         token_delta *= self.baseband_weight
#                     if is_image:
#                         token_delta *= self.image_int

#                     heatmap_band_maps[bb] = token_delta.clamp_min(0.)

#                 embeddings = embeddings_tokens

#                 if self.pool_type == 'avg':
#                     embeddings = embeddings.mean(dim=2)
#                 elif self.pool_type == 'max':
#                     embeddings, _ = embeddings.max(dim=2)
#                 else:
#                     raise ValueError(f"Unsupported pooling type: {self.pool_type}")

#             elif f_d.dim() == 4:
#                 batch, height, width, _ = f_d.shape
#                 frames = 1
#                 f_d_flat = f_d.reshape(-1, 8)
#                 embeddings = self.feature_net(f_d_flat).reshape(batch, height * width, self.embedding_dim)

#                 if self.pool_type == 'avg':
#                     embeddings = embeddings.mean(dim=1, keepdim=True)
#                 elif self.pool_type == 'max':
#                     embeddings, _ = embeddings.max(dim=1, keepdim=True)
#                 else:
#                     raise ValueError(f"Unsupported pooling type: {self.pool_type}")

#                 if return_heatmap:
#                     dino_cur = dino_ref
#                     if dino_cur.dim() == 2:
#                         dino_cur = dino_cur.unsqueeze(0)
#                     if dino_cur.shape[0] == 1 and batch > 1:
#                         dino_cur = dino_cur.expand(batch, -1, -1)
#                     if dino_cur.shape[1] == 1:
#                         dino_cur = dino_cur.expand(-1, frames, -1)

#                     dino_map = dino_cur[:, :, None, :].expand(-1, -1, height * width, -1)
#                     tokens = self.feature_net(f_d_flat).reshape(batch, frames, height * width, self.embedding_dim)
#                     combined_tokens = torch.cat((tokens, dino_map), dim=-1)
#                     token_delta = self.quality_net(combined_tokens.reshape(-1, self.embedding_dim + 768)).reshape(batch, frames, height, width)

#                     is_base_band = (bb == no_bands - 1)
#                     if is_base_band:
#                         token_delta *= self.baseband_weight
#                     if is_image:
#                         token_delta *= self.image_int

#                     heatmap_band_maps[bb] = token_delta.clamp_min(0.)
#             else:
#                 raise RuntimeError(f"Unsupported feature tensor dimensionality: {f_d.dim()}")

#             band_embeddings.append(embeddings)

#         band_embeddings = torch.stack(band_embeddings, dim=2)

#         baseband_weight = self.baseband_weight
#         if isinstance(baseband_weight, torch.Tensor):
#             if baseband_weight.numel() > 1:
#                 baseband_weight = baseband_weight.mean()
#             baseband_weight = baseband_weight.to(self.device)

#         band_weights = torch.ones((1, 1, no_bands, 1), device=self.device, dtype=band_embeddings.dtype)
#         band_weights[:, :, no_bands - 1, :] = baseband_weight
#         weighted_band_embeddings = band_embeddings * band_weights

#         if self.pool_type == 'avg':
#             pooled_embeddings = weighted_band_embeddings.sum(dim=2) / band_weights.sum(dim=2).clamp_min(1e-8)
#         elif self.pool_type == 'max':
#             pooled_embeddings, _ = weighted_band_embeddings.max(dim=2)
#         else:
#             raise ValueError(f"Unsupported pooling type: {self.pool_type}")

#         batch, frames, _ = pooled_embeddings.shape

#         if dino_ref.dim() == 2:
#             if dino_ref.shape[0] == frames:
#                 dino_ref = dino_ref.unsqueeze(0)
#             else:
#                 dino_ref = dino_ref.unsqueeze(1)

#         if dino_ref.dim() == 3 and dino_ref.shape[0] == 1 and batch > 1:
#             dino_ref = dino_ref.expand(batch, -1, -1)

#         if dino_ref.shape[1] != frames:
#             if dino_ref.shape[1] == 1:
#                 dino_ref = dino_ref.expand(-1, frames, -1)
#             else:
#                 min_frames = min(dino_ref.shape[1], frames)
#                 dino_ref = dino_ref[:, :min_frames, :]
#                 pooled_embeddings = pooled_embeddings[:, :min_frames, :]
#                 frames = min_frames

#         combined_features = torch.cat((pooled_embeddings, dino_ref), dim=-1)
#         combined_flat = combined_features.reshape(-1, self.embedding_dim + 768)
#         delta = self.quality_net(combined_flat)

#         delta = delta.reshape(batch, frames)
#         delta = delta.mean(dim=-1).mean()

#         if is_image:
#             delta *= self.image_int

#         q_jod = q_jod - delta

#         assert not q_jod.isnan()
#         if not return_heatmap:
#             return q_jod

#         heatmap_pyr = lpyr_dec_2(self.lpyr.W, self.lpyr.H, self.pix_per_deg, self.device)
#         heatmap_pyr.decompose(torch.zeros((1, 1, self.lpyr.H, self.lpyr.W), dtype=features[0].dtype, device=self.device))

#         for bb in range(no_bands):
#             band_map = heatmap_band_maps[bb]
#             target_h = heatmap_pyr.get_lband(bb).shape[-2]
#             target_w = heatmap_pyr.get_lband(bb).shape[-1]

#             if band_map.shape[-2] != target_h or band_map.shape[-1] != target_w:
#                 bsz, nframes, src_h, src_w = band_map.shape
#                 band_map = Func.adaptive_avg_pool2d(
#                     band_map.reshape(bsz * nframes, 1, src_h, src_w),
#                     output_size=(target_h, target_w)
#                 ).reshape(bsz, nframes, target_h, target_w)

#             heatmap_band_maps[bb] = band_map

#         n_frames = heatmap_band_maps[0].shape[1]
#         heatmap_raw = torch.empty((batch, n_frames, self.lpyr.H, self.lpyr.W), dtype=features[0].dtype, device=self.device)

#         with torch.no_grad():
#             for bidx in range(batch):
#                 for ff in range(n_frames):
#                     frame_pyr = lpyr_dec_2(self.lpyr.W, self.lpyr.H, self.pix_per_deg, self.device)
#                     frame_pyr.decompose(torch.zeros((1, 1, self.lpyr.H, self.lpyr.W), dtype=features[0].dtype, device=self.device))
#                     for bb in range(no_bands):
#                         frame_pyr.set_lband(bb, heatmap_band_maps[bb][bidx:bidx+1, ff:ff+1, :, :])
#                     frame_map = 1. - (self.met2jod(frame_pyr.reconstruct()) / 10.)
#                     heatmap_raw[bidx:bidx+1, ff:ff+1, :, :] = frame_map

#         return q_jod, heatmap_raw


# register_metric(cvvdp_ml_dino_fusion_simplified)


# class cvvdp_ml_dino_fusion_simplified_v2(cvvdp_ml_dino_fusion_simplified):
#     """
#     Simplified DINO-fusion v2 with a small hidden-layer quality head.
#     """

#     def __init__(self, device=None, pool_type='avg', quality_hidden_dim=256, **kwargs):
#         self.set_device(device)

#         embedding_dim = 128
#         self.feature_net = nn.Linear(8, embedding_dim).to(self.device)

#         mlp_in_channels = embedding_dim + 768
#         self.quality_net = nn.Sequential(
#             nn.LayerNorm(mlp_in_channels),
#             MLP(
#                 in_channels=mlp_in_channels,
#                 hidden_channels=[256, 128, 1],
#                 activation_layer=torch.nn.ReLU,
#                 dropout=0.1,
#             ),
#         ).to(self.device)

#         self.pool_type = pool_type
#         self.embedding_dim = embedding_dim
#         self.supports_heatmap = True

#         cvvdp_ml_dino_base.__init__(self, device=device, **kwargs)

#         self.train(False)


# register_metric(cvvdp_ml_dino_fusion_simplified_v2)


# class cvvdp_ml_dino_attention(cvvdp_ml_dino_base):

#     def __init__(self, device=None, **kwargs):
#         self.set_device(device)

#         self.embedding_dim = 128
#         dino_dim = 768

#         self.feature_net = nn.Sequential(
#             nn.Linear(8, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, self.embedding_dim),
#         ).to(self.device)

#         self.query_proj = nn.Sequential(
#             nn.Linear(dino_dim, self.embedding_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#         ).to(self.device)

#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=self.embedding_dim,
#             num_heads=8,
#             dropout=0.1,
#             batch_first=True,
#         ).to(self.device)

#         self.attn_norm = nn.LayerNorm(self.embedding_dim).to(self.device)

#         self.quality_net = MLP(
#             in_channels=self.embedding_dim,
#             hidden_channels=[384, 128, 1],
#             activation_layer=torch.nn.ReLU,
#             dropout=0.2,
#         ).to(self.device)

#         super().__init__(device=device, **kwargs)

#         self.train(False)

#     def get_nets_to_load(self):
#         return ['feature_net', 'query_proj', 'cross_attn', 'attn_norm', 'quality_net']

#     def do_pooling_and_jods(self, features, dino_features):
#         no_bands = len(features)
#         q_jod = torch.as_tensor(10., device=self.device)

#         baseband_weight = self.baseband_weight
#         if isinstance(baseband_weight, torch.Tensor):
#             if baseband_weight.numel() > 1:
#                 baseband_weight = baseband_weight.mean()
#             baseband_weight = baseband_weight.to(self.device)

#         f0 = features[0]
#         ch_dim = 4 if f0.dim() == 6 else 3
#         is_image = (f0.shape[ch_dim] == 3)

#         band_tokens = []
#         for bb in range(no_bands):
#             f = features[bb]

#             if f.dim() == 4:
#                 f = f.unsqueeze(0)

#             if is_image:
#                 f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], f.shape[3], 1, f.shape[5]), device=self.device)), dim=4)

#             if self.disabled_features is not None:
#                 f[..., self.disabled_features] = 0

#             f_d = f[..., 4:]
#             f_d[..., 1] = torch.sqrt(torch.abs(f_d[..., 1]))
#             f_d = f_d.flatten(start_dim=f_d.dim() - 2)

#             bsz, nframes, hsz, wsz, _ = f_d.shape
#             f_d_flat = f_d.reshape(-1, 8)
#             emb = self.feature_net(f_d_flat).reshape(bsz, nframes, hsz * wsz, self.embedding_dim)

#             band_tokens.append(emb)

#         token_bank = torch.cat(band_tokens, dim=2)

#         dino_ref = dino_features[0].to(self.device)
#         if dino_ref.dim() == 2:
#             dino_ref = dino_ref.unsqueeze(0)

#         if dino_ref.shape[0] == 1 and token_bank.shape[0] > 1:
#             dino_ref = dino_ref.expand(token_bank.shape[0], -1, -1)

#         if dino_ref.shape[1] != token_bank.shape[1]:
#             if dino_ref.shape[1] == 1:
#                 dino_ref = dino_ref.expand(-1, token_bank.shape[1], -1)
#             else:
#                 min_frames = min(dino_ref.shape[1], token_bank.shape[1])
#                 dino_ref = dino_ref[:, :min_frames, :]
#                 token_bank = token_bank[:, :min_frames, :, :]

#         bsz, nframes, ntokens, _ = token_bank.shape
#         query = self.query_proj(dino_ref).reshape(bsz * nframes, 1, self.embedding_dim)
#         key_value = token_bank.reshape(bsz * nframes, ntokens, self.embedding_dim)

#         attn_out, _ = self.cross_attn(query, key_value, key_value, need_weights=False)
#         attn_out = self.attn_norm(attn_out + query).reshape(bsz, nframes, self.embedding_dim)

#         delta = self.quality_net(attn_out.reshape(-1, self.embedding_dim)).reshape(bsz, nframes)

#         if is_image:
#             delta *= self.image_int

#         q_jod -= delta.mean(dim=1).mean()

#         assert not q_jod.isnan()
#         return q_jod


# register_metric(cvvdp_ml_dino_attention)


# class cvvdp_ml_dino_attention_v2(cvvdp_ml_dino_base):

#     def __init__(self, device=None, dim=128, max_bands=16, **kwargs):
#         self.set_device(device)

#         self.embedding_dim = dim
#         self.max_bands = max_bands
#         dino_dim = 768

#         self.feature_net = nn.Sequential(
#             nn.Linear(8, 256),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, self.embedding_dim),
#         ).to(self.device)

#         self.spatial_pos_mlp = nn.Sequential(
#             nn.Linear(2, self.embedding_dim // 2),
#             nn.GELU(),
#             nn.Linear(self.embedding_dim // 2, self.embedding_dim)
#         ).to(self.device)

#         self.band_pos_embed = nn.Embedding(self.max_bands, self.embedding_dim).to(self.device)
#         self.baseband_flag_embed = nn.Embedding(2, self.embedding_dim).to(self.device)

#         self.query_proj = nn.Sequential(
#             nn.Linear(dino_dim, self.embedding_dim),
#             nn.GELU(),
#             nn.Dropout(0.2),
#         ).to(self.device)

#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=self.embedding_dim,
#             num_heads=8,
#             dropout=0.1,
#             batch_first=True,
#         ).to(self.device)

#         self.attn_norm = nn.LayerNorm(self.embedding_dim).to(self.device)

#         self.quality_net = MLP(
#             in_channels=self.embedding_dim,
#             hidden_channels=[256, 64, 1],
#             activation_layer=torch.nn.ReLU,
#             dropout=0.2,
#         ).to(self.device)

#         super().__init__(device=device, **kwargs)

#         self.train(False)

#     def get_nets_to_load(self):
#         return [
#             'feature_net',
#             'spatial_pos_mlp',
#             'band_pos_embed',
#             'baseband_flag_embed',
#             'query_proj',
#             'cross_attn',
#             'attn_norm',
#             'quality_net'
#         ]

#     def _get_spatial_position_embedding(self, h, w, device):
#         y_coords = (torch.arange(h, device=device).float() + 0.5) / h
#         x_coords = (torch.arange(w, device=device).float() + 0.5) / w
#         grid = torch.stack(torch.meshgrid(y_coords, x_coords, indexing='ij'), dim=-1)
#         pos_embed = self.spatial_pos_mlp(grid)
#         return pos_embed.view(1, 1, h * w, self.embedding_dim)

#     def do_pooling_and_jods(self, features, dino_features):
#         no_bands = len(features)
#         if no_bands > self.max_bands:
#             raise RuntimeError(f"Number of bands ({no_bands}) exceeds max_bands ({self.max_bands})")

#         q_jod = torch.as_tensor(10., device=self.device)

#         f0 = features[0]
#         ch_dim = 4 if f0.dim() == 6 else 3
#         is_image = (f0.shape[ch_dim] == 3)

#         band_tokens = []
#         for bb in range(no_bands):
#             f = features[bb]

#             if f.dim() == 4:
#                 f = f.unsqueeze(0)

#             if is_image:
#                 f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], f.shape[3], 1, f.shape[5]), device=self.device)), dim=4)

#             if self.disabled_features is not None:
#                 f[..., self.disabled_features] = 0

#             f_d = f[..., 4:]
#             f_d[..., 1] = torch.sqrt(torch.abs(f_d[..., 1]))
#             f_d = f_d.flatten(start_dim=f_d.dim() - 2)

#             bsz, nframes, hsz, wsz, _ = f_d.shape
#             emb = self.feature_net(f_d.reshape(-1, 8)).reshape(bsz, nframes, hsz * wsz, self.embedding_dim)

#             spatial_pos = self._get_spatial_position_embedding(hsz, wsz, f.device)
#             emb = emb + spatial_pos

#             band_index = torch.as_tensor([bb], dtype=torch.long, device=f.device)
#             band_pos = self.band_pos_embed(band_index).view(1, 1, 1, self.embedding_dim)
#             emb = emb + band_pos

#             is_baseband = 1 if bb == (no_bands - 1) else 0
#             baseband_index = torch.as_tensor([is_baseband], dtype=torch.long, device=f.device)
#             baseband_pos = self.baseband_flag_embed(baseband_index).view(1, 1, 1, self.embedding_dim)
#             emb = emb + baseband_pos

#             band_tokens.append(emb)

#         token_bank = torch.cat(band_tokens, dim=2)

#         dino_ref = dino_features[0].to(self.device)
#         if dino_ref.dim() == 2:
#             dino_ref = dino_ref.unsqueeze(0)

#         if dino_ref.shape[0] == 1 and token_bank.shape[0] > 1:
#             dino_ref = dino_ref.expand(token_bank.shape[0], -1, -1)

#         if dino_ref.shape[1] != token_bank.shape[1]:
#             if dino_ref.shape[1] == 1:
#                 dino_ref = dino_ref.expand(-1, token_bank.shape[1], -1)
#             else:
#                 min_frames = min(dino_ref.shape[1], token_bank.shape[1])
#                 dino_ref = dino_ref[:, :min_frames, :]
#                 token_bank = token_bank[:, :min_frames, :, :]

#         bsz, nframes, ntokens, _ = token_bank.shape
#         query = self.query_proj(dino_ref).reshape(bsz * nframes, 1, self.embedding_dim)
#         key_value = token_bank.reshape(bsz * nframes, ntokens, self.embedding_dim)

#         attn_out, _ = self.cross_attn(query, key_value, key_value, need_weights=False)
#         attn_out = self.attn_norm(attn_out + query).reshape(bsz, nframes, self.embedding_dim)

#         delta = self.quality_net(attn_out.reshape(-1, self.embedding_dim)).reshape(bsz, nframes)

#         if is_image:
#             delta *= self.image_int

#         q_jod -= delta.mean(dim=1).mean()

#         assert not q_jod.isnan()
#         return q_jod


# register_metric(cvvdp_ml_dino_attention_v2)


# class cvvdp_ml_dino_attention_hybrid(cvvdp_ml_dino_base):

#     def __init__(self, device=None, dim=128, max_bands=16, tokens_per_band=16, use_positional=True, **kwargs):
#         self.set_device(device)

#         self.embedding_dim = dim
#         self.max_bands = max_bands
#         self.tokens_per_band = tokens_per_band
#         self.use_positional = use_positional
#         dino_dim = 768

#         self.feature_net = nn.Sequential(
#             nn.Linear(8, 256),
#             nn.GELU(),
#             nn.Dropout(0.2),
#             nn.Linear(256, self.embedding_dim),
#         ).to(self.device)

#         if self.use_positional:
#             self.spatial_pos_mlp = nn.Sequential(
#                 nn.Linear(2, self.embedding_dim // 2),
#                 nn.GELU(),
#                 nn.Linear(self.embedding_dim // 2, self.embedding_dim)
#             ).to(self.device)

#             self.band_pos_embed = nn.Embedding(self.max_bands, self.embedding_dim).to(self.device)
#             self.baseband_flag_embed = nn.Embedding(2, self.embedding_dim).to(self.device)

#             self.spatial_pos_scale = nn.Parameter(torch.tensor(1.0, device=self.device))
#             self.band_pos_scale = nn.Parameter(torch.tensor(1.0, device=self.device))
#             self.baseband_pos_scale = nn.Parameter(torch.tensor(1.0, device=self.device))

#         self.query_proj = nn.Sequential(
#             nn.Linear(dino_dim, self.embedding_dim),
#             nn.GELU(),
#             nn.Dropout(0.2),
#         ).to(self.device)

#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=self.embedding_dim,
#             num_heads=8,
#             dropout=0.1,
#             batch_first=True,
#         ).to(self.device)

#         self.attn_norm = nn.LayerNorm(self.embedding_dim).to(self.device)

#         self.quality_net = MLP(
#             in_channels=self.embedding_dim,
#             hidden_channels=[384, 128, 1],
#             activation_layer=torch.nn.ReLU,
#             dropout=0.2,
#         ).to(self.device)

#         super().__init__(device=device, **kwargs)

#         self.train(False)

#     def get_nets_to_load(self):
#         nets = [
#             'feature_net',
#             'query_proj',
#             'cross_attn',
#             'attn_norm',
#             'quality_net'
#         ]
#         if self.use_positional:
#             nets.extend([
#                 'spatial_pos_mlp',
#                 'band_pos_embed',
#                 'baseband_flag_embed'
#             ])
#         return nets

#     def _get_spatial_position_embedding(self, h, w, device):
#         y_coords = (torch.arange(h, device=device).float() + 0.5) / h
#         x_coords = (torch.arange(w, device=device).float() + 0.5) / w
#         grid = torch.stack(torch.meshgrid(y_coords, x_coords, indexing='ij'), dim=-1)
#         pos_embed = self.spatial_pos_mlp(grid)
#         return pos_embed.view(1, 1, h * w, self.embedding_dim)

#     def _reduce_spatial_tokens(self, emb, hsz, wsz):
#         if self.tokens_per_band is None or self.tokens_per_band <= 0:
#             return emb

#         max_tokens = hsz * wsz
#         if self.tokens_per_band >= max_tokens:
#             return emb

#         pool_h = max(1, int(math.sqrt(self.tokens_per_band)))
#         pool_w = max(1, int(math.ceil(self.tokens_per_band / pool_h)))

#         bsz, nframes, _, emb_dim = emb.shape
#         emb_map = emb.reshape(bsz * nframes, hsz, wsz, emb_dim).permute(0, 3, 1, 2)
#         pooled = Func.adaptive_avg_pool2d(emb_map, output_size=(pool_h, pool_w))
#         pooled = pooled.permute(0, 2, 3, 1).reshape(bsz, nframes, pool_h * pool_w, emb_dim)

#         if pooled.shape[2] > self.tokens_per_band:
#             pooled = pooled[:, :, :self.tokens_per_band, :]

#         return pooled

#     def do_pooling_and_jods(self, features, dino_features):
#         no_bands = len(features)
#         if no_bands > self.max_bands:
#             raise RuntimeError(f"Number of bands ({no_bands}) exceeds max_bands ({self.max_bands})")

#         q_jod = torch.as_tensor(10., device=self.device)

#         baseband_weight = self.baseband_weight
#         if isinstance(baseband_weight, torch.Tensor):
#             if baseband_weight.numel() > 1:
#                 baseband_weight = baseband_weight.mean()
#             baseband_weight = baseband_weight.to(self.device)

#         f0 = features[0]
#         ch_dim = 4 if f0.dim() == 6 else 3
#         is_image = (f0.shape[ch_dim] == 3)

#         band_tokens = []
#         for bb in range(no_bands):
#             f = features[bb]

#             if f.dim() == 4:
#                 f = f.unsqueeze(0)

#             if is_image:
#                 f = torch.cat((f, torch.zeros((f.shape[0], f.shape[1], f.shape[2], f.shape[3], 1, f.shape[5]), device=self.device)), dim=4)

#             if self.disabled_features is not None:
#                 f[..., self.disabled_features] = 0

#             f_d = f[..., 4:]
#             f_d[..., 1] = torch.sqrt(torch.abs(f_d[..., 1]))
#             f_d = f_d.flatten(start_dim=f_d.dim() - 2)

#             bsz, nframes, hsz, wsz, _ = f_d.shape
#             emb = self.feature_net(f_d.reshape(-1, 8)).reshape(bsz, nframes, hsz * wsz, self.embedding_dim)

#             if self.use_positional:
#                 spatial_pos = self._get_spatial_position_embedding(hsz, wsz, f.device)
#                 emb = emb + self.spatial_pos_scale * spatial_pos

#                 band_index = torch.as_tensor([bb], dtype=torch.long, device=f.device)
#                 band_pos = self.band_pos_embed(band_index).view(1, 1, 1, self.embedding_dim)
#                 emb = emb + self.band_pos_scale * band_pos

#                 is_baseband = 1 if bb == (no_bands - 1) else 0
#                 baseband_index = torch.as_tensor([is_baseband], dtype=torch.long, device=f.device)
#                 baseband_pos = self.baseband_flag_embed(baseband_index).view(1, 1, 1, self.embedding_dim)
#                 emb = emb + self.baseband_pos_scale * baseband_pos

#             if bb == (no_bands - 1):
#                 emb = emb * baseband_weight

#             emb = self._reduce_spatial_tokens(emb, hsz, wsz)
#             band_tokens.append(emb)

#         token_bank = torch.cat(band_tokens, dim=2)

#         dino_ref = dino_features[0].to(self.device)
#         if dino_ref.dim() == 2:
#             dino_ref = dino_ref.unsqueeze(0)

#         if dino_ref.shape[0] == 1 and token_bank.shape[0] > 1:
#             dino_ref = dino_ref.expand(token_bank.shape[0], -1, -1)

#         if dino_ref.shape[1] != token_bank.shape[1]:
#             if dino_ref.shape[1] == 1:
#                 dino_ref = dino_ref.expand(-1, token_bank.shape[1], -1)
#             else:
#                 min_frames = min(dino_ref.shape[1], token_bank.shape[1])
#                 dino_ref = dino_ref[:, :min_frames, :]
#                 token_bank = token_bank[:, :min_frames, :, :]

#         bsz, nframes, ntokens, _ = token_bank.shape
#         query = self.query_proj(dino_ref).reshape(bsz * nframes, 1, self.embedding_dim)
#         key_value = token_bank.reshape(bsz * nframes, ntokens, self.embedding_dim)

#         attn_out, _ = self.cross_attn(query, key_value, key_value, need_weights=False)
#         attn_out = self.attn_norm(attn_out + query).reshape(bsz, nframes, self.embedding_dim)

#         delta = self.quality_net(attn_out.reshape(-1, self.embedding_dim)).reshape(bsz, nframes)

#         if is_image:
#             delta *= self.image_int

#         q_jod -= delta.mean(dim=1).mean()

#         assert not q_jod.isnan()
#         return q_jod


# register_metric(cvvdp_ml_dino_attention_hybrid)

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


class cvvdp_ml_saliency_v2(cvvdp_ml_saliency):

    def __init__(self,
                 config_paths=[],
                 device=None,
                 saliency_norm_eps=1e-8,
                 positive_floor=1e-6,
                 distortion_weight_alpha=1.0,
                 distortion_weight_eps=1e-8,
                 distortion_weight_power=1.0,
                 distortion_weight_max=5.0,
                 **kwargs):

        self.saliency_norm_eps = saliency_norm_eps
        self.positive_floor = positive_floor
        self.distortion_weight_alpha = distortion_weight_alpha
        self.distortion_weight_eps = distortion_weight_eps
        self.distortion_weight_power = distortion_weight_power
        self.distortion_weight_max = distortion_weight_max
        super().__init__(config_paths=config_paths, device=device, **kwargs)

    def _normalize_across_band_patches(self, weights):
        eps = torch.as_tensor(self.saliency_norm_eps, dtype=weights.dtype, device=weights.device).clamp_min(1e-12)
        denom = weights.sum(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
        patch_count = float(weights.shape[1] * weights.shape[2] * weights.shape[3])
        return (weights / denom) * patch_count

    def _compute_distortion_weight(self, f):
        eps = torch.as_tensor(self.distortion_weight_eps, dtype=f.dtype, device=f.device).clamp_min(1e-12)
        power = torch.as_tensor(self.distortion_weight_power, dtype=f.dtype, device=f.device).clamp_min(0.0)
        alpha = torch.as_tensor(self.distortion_weight_alpha, dtype=f.dtype, device=f.device).clamp(0.0, 1.0)

        local_dist = torch.mean(torch.abs(f[..., 4]), dim=4, keepdim=True)
        local_dist = torch.pow(local_dist + eps, power)
        dist_mean = local_dist.mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
        dist_weight = local_dist / dist_mean

        if self.distortion_weight_max is not None:
            max_weight = torch.as_tensor(self.distortion_weight_max, dtype=f.dtype, device=f.device).clamp_min(1.0)
            dist_weight = torch.clamp(dist_weight, max=max_weight)

        dist_weight = (1.0 - alpha) + alpha * dist_weight

        return dist_weight

    def do_pooling_and_jods(self, features):
        no_bands = len(features)
        batch_sz = features[0].shape[0]

        Q_JOD = torch.ones((batch_sz), device=self.device) * 10.

        is_image = (features[0].shape[4] == 3)

        for bb in range(no_bands):
            f = features[bb]

            f[..., 1::2] = torch.sqrt(torch.abs(f[..., 1::2]))

            if is_image:
                f = torch.cat((f, torch.zeros((f.shape[0:4] + (1, f.shape[5])), device=self.device)), dim=4)
            if self.disabled_features is not None:
                f[..., self.disabled_features] = 0

            f_TR = f[..., 0:4].flatten(start_dim=4)
            f_D = f[..., 4:].flatten(start_dim=4)

            att = F.softplus(self.att_net(f_TR)) + self.positive_floor
            att = self._normalize_across_band_patches(att)

            dist_weight = self._compute_distortion_weight(f)

            D_all = F.softplus(self.feature_net(f_D)) + self.positive_floor
            D_all = D_all * att * dist_weight / no_bands

            is_base_band = (bb == no_bands - 1)
            if is_base_band:
                D_all *= self.baseband_weight

            if is_image:
                D_all *= self.image_int

            Q_JOD -= self.spatiotemporal_pooling(D_all)

        assert(not Q_JOD.isnan().any())
        return Q_JOD

    def full_name(self):
        return "ColorVideoVDP-ML-Saliency-v2"


register_metric(cvvdp_ml_saliency_v2)


class cvvdp_ml_saliency_plus(cvvdp_ml_saliency_base):

    def __init__(self, config_paths=[], device=None, saliency_min=1e-4, saliency_band_smoothing=True, **kwargs):

        self.set_device(device)

        dropout = 0.2
        hidden_dims = 24
        num_layers = 3
        ch_no = 4
        stats_no = 2
        self.feature_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

        self.saliency_min = saliency_min
        self.saliency_band_smoothing = saliency_band_smoothing
        self.supports_heatmap = True

        super().__init__(config_paths=config_paths, device=device, **kwargs)

    def get_nets_to_load(self):
        return ['feature_net']

    def _compute_band_saliency(self, saliency_maps, target_h, target_w, band_index, no_bands):
        batch_sz, no_frames, src_h, src_w = saliency_maps.shape
        sal_map = saliency_maps.reshape(batch_sz * no_frames, 1, src_h, src_w)

        if self.saliency_band_smoothing and no_bands > 1:
            norm_pos = float(band_index) / float(no_bands - 1)
            smooth_steps = int(round(norm_pos * 3.0))
            kernel = 1 + 2 * smooth_steps
            if kernel > 1:
                sal_map = Func.avg_pool2d(sal_map, kernel_size=kernel, stride=1, padding=kernel // 2)

        sal_map = Func.adaptive_avg_pool2d(sal_map, output_size=(target_h, target_w))
        sal_map = sal_map.reshape(batch_sz, no_frames, target_h, target_w)

        sal_map = sal_map / sal_map.mean(dim=(2, 3), keepdim=True).clamp_min(self.saliency_min)
        sal_map = sal_map.clamp_min(self.saliency_min)
        return sal_map

    def _extract_reference_context(self, vid_source, met_colorspace):
        _, _, no_frames = vid_source.get_video_size()

        ref_frames = []
        for ff in range(no_frames):
            ref_frame = vid_source.get_reference_frame(ff, device=self.device, colorspace=met_colorspace).squeeze(2)
            ref_frames.append(ref_frame[:, 0, :, :])

        return torch.stack(ref_frames, dim=1)

    def _clone_video_source(self, vid_source):
        if not hasattr(vid_source, 'test_fname') or not hasattr(vid_source, 'reference_fname'):
            return None

        ctor_kwargs = {
            'test_fname': vid_source.test_fname,
            'reference_fname': vid_source.reference_fname,
            'display_photometry': self.display_photometry,
            'config_paths': getattr(self, 'config_paths', []),
            'fps': getattr(vid_source, 'fps', None),
            'frames': getattr(vid_source, 'in_frames', -1),
            'full_screen_resize': getattr(vid_source, 'full_screen_resize', None),
            'resize_resolution': getattr(vid_source, 'resize_resolution', None),
            'ffmpeg_cc': getattr(vid_source, 'ffmpeg_cc', False),
            'verbose': getattr(vid_source, 'verbose', False),
        }

        try:
            return vid_source.__class__(**ctor_kwargs)
        except TypeError:
            try:
                return vid_source.__class__(
                    vid_source.test_fname,
                    vid_source.reference_fname,
                    display_photometry=self.display_photometry,
                    config_paths=getattr(self, 'config_paths', []),
                    fps=getattr(vid_source, 'fps', None),
                    frames=getattr(vid_source, 'in_frames', -1),
                    full_screen_resize=getattr(vid_source, 'full_screen_resize', None),
                    resize_resolution=getattr(vid_source, 'resize_resolution', None),
                    ffmpeg_cc=getattr(vid_source, 'ffmpeg_cc', False),
                    verbose=getattr(vid_source, 'verbose', False),
                )
            except Exception:
                return None

    def predict_video_source(self, vid_source):
        assert vid_source.get_batch_size()==1 or self.heatmap is None or self.heatmap=='none', 'Heatmaps not supported when batches are used'

        saliency_vid_source = self._clone_video_source(vid_source)
        context_vid_source = self._clone_video_source(vid_source) if (self.do_heatmap and self.heatmap != 'raw') else None

        if saliency_vid_source is None:
            saliency_features = self.extract_features_saliency(vid_source)
        else:
            saliency_features = self.extract_features_saliency(saliency_vid_source)

        use_ml_heatmap = self.do_heatmap
        prev_do_heatmap = self.do_heatmap

        if use_ml_heatmap:
            self.do_heatmap = False

        try:
            features, _ = self.extract_features(vid_source)
        finally:
            self.do_heatmap = prev_do_heatmap

        if use_ml_heatmap:
            q_jod, heatmap_raw = self.do_pooling_and_jods(features, saliency_features, return_heatmap=True)
            if self.heatmap == 'raw':
                heatmap = heatmap_raw.detach().type(torch.float16).cpu().unsqueeze(1)
            else:
                met_colorspace = 'logLMS_DKLd65' if self.contrast == 'log' else 'DKLd65'
                context_source = context_vid_source if context_vid_source is not None else vid_source
                ref_context = self._extract_reference_context(context_source, met_colorspace).detach().cpu()
                heatmap_raw_cpu = heatmap_raw.detach().cpu()
                heatmap = visualize_diff_map(
                    heatmap_raw_cpu,
                    context_image=ref_context,
                    colormap_type=self.heatmap,
                    use_cpu=False
                ).detach().type(torch.float16).cpu().unsqueeze(0)
        else:
            q_jod = self.do_pooling_and_jods(features, saliency_features)
            heatmap = None

        vid_sz = vid_source.get_video_size()
        height, width, n_frames = vid_sz

        stats = {}
        stats['rho_band'] = self.lpyr.get_freqs()
        stats['frames_per_second'] = vid_source.get_frames_per_second()
        stats['width'] = width
        stats['height'] = height
        stats['N_frames'] = n_frames

        if self.dump_channels:
            self.dump_channels.close()

        if use_ml_heatmap:
            stats['heatmap'] = heatmap

        return (q_jod.squeeze(), stats)

    def do_pooling_and_jods(self, features, saliency_features, return_heatmap=False):

        no_bands = len(features)
        batch_sz = features[0].shape[0]

        Q_JOD = torch.ones((batch_sz), device=self.device) * 10.

        is_image = (features[0].shape[4] == 3)

        heatmap_pyr = None
        heatmap_band_maps = None
        if return_heatmap:
            heatmap_pyr = lpyr_dec_2(self.lpyr.W, self.lpyr.H, self.pix_per_deg, self.device)
            heatmap_band_maps = [None] * no_bands
            dummy_heatmap = torch.zeros(
                (1, 1, self.lpyr.H, self.lpyr.W),
                dtype=features[0].dtype,
                device=self.device
            )
            heatmap_pyr.decompose(dummy_heatmap)

        saliency_maps = saliency_features[0].to(self.device)
        if saliency_maps.dim() == 3:
            saliency_maps = saliency_maps[None, ...]

        if saliency_maps.dim() != 4:
            raise RuntimeError(f"Expected saliency_features[0] to have 4 dims [B,F,H,W], got {saliency_maps.shape}")

        for bb in range(no_bands):
            f = features[bb]

            f[..., 1::2] = torch.sqrt(torch.abs(f[..., 1::2]))

            if is_image:
                f = torch.cat((f, torch.zeros((f.shape[0:4] + (1, f.shape[5])), device=self.device)), dim=4)
            if self.disabled_features is not None:
                f[..., self.disabled_features] = 0

            target_h, target_w = f.shape[2], f.shape[3]
            band_saliency = self._compute_band_saliency(saliency_maps, target_h, target_w, bb, no_bands)

            f_D = f[..., 4:] * band_saliency[..., None, None]
            f_D = f_D.flatten(start_dim=4)

            D_all = self.feature_net(f_D)
            D_all = F.relu(D_all) / no_bands

            is_base_band = (bb == no_bands - 1)
            if is_base_band:
                D_all *= self.baseband_weight

            if is_image:
                D_all *= self.image_int

            if return_heatmap:
                heatmap_band_maps[bb] = D_all.squeeze(-1)

            Q_JOD -= self.spatiotemporal_pooling(D_all)

        assert(not Q_JOD.isnan().any())
        if not return_heatmap:
            return Q_JOD

        for bb in range(no_bands):
            band_map = heatmap_band_maps[bb]
            target_h = heatmap_pyr.get_lband(bb).shape[-2]
            target_w = heatmap_pyr.get_lband(bb).shape[-1]

            if band_map.shape[-2] != target_h or band_map.shape[-1] != target_w:
                bsz, nframes, src_h, src_w = band_map.shape
                band_map_resized = Func.adaptive_avg_pool2d(
                    band_map.reshape(bsz * nframes, 1, src_h, src_w),
                    output_size=(target_h, target_w)
                ).reshape(bsz, nframes, target_h, target_w)
            else:
                band_map_resized = band_map

            heatmap_band_maps[bb] = band_map_resized

        n_frames = heatmap_band_maps[0].shape[1]
        heatmap_raw = torch.empty((batch_sz, n_frames, self.lpyr.H, self.lpyr.W), dtype=features[0].dtype, device=self.device)

        with torch.no_grad():
            for bidx in range(batch_sz):
                for ff in range(n_frames):
                    frame_pyr = lpyr_dec_2(self.lpyr.W, self.lpyr.H, self.pix_per_deg, self.device)
                    frame_pyr.decompose(torch.zeros((1, 1, self.lpyr.H, self.lpyr.W), dtype=features[0].dtype, device=self.device))
                    for bb in range(no_bands):
                        frame_pyr.set_lband(bb, heatmap_band_maps[bb][bidx:bidx+1, ff:ff+1, :, :])
                    frame_map = 1. - (self.met2jod(frame_pyr.reconstruct()) / 10.)
                    heatmap_raw[bidx:bidx+1, ff:ff+1, :, :] = frame_map

        return Q_JOD, heatmap_raw

    def full_name(self):
        return "ColorVideoVDP-ML-Saliency-Plus"

    def spatiotemporal_pooling(self, D_all):
        return D_all.view(D_all.shape[0], -1).mean(dim=1)


register_metric(cvvdp_ml_saliency_plus)


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
        # x: [B, D, H, W, C]
        
        B, D, H, W, C = x.shape
        x = x.reshape(B * D, H, W, C)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.patch_embed(x)  # [B, N_patches, dim]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        cls_feat = x[:, 0]
        y = self.reg_head(cls_feat).squeeze(-1).reshape(B,D)
        return y.mean(dim=1, keepdim=False)
    
    def get_heatmap(self, x):
        # x: [B, D, H, W, C]
        B, D, H, W, C = x.shape
        x = x.reshape(B * D, H, W, C)
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
        batch_sz = features[0].shape[0]
        Q_JOD = torch.ones((batch_sz), device=self.device)*10.
        is_image = (features[0].shape[4]==3) # if 3 channels, it is an image

        for bb, f in enumerate(features):

            f[..., 1::2] = torch.sqrt(torch.abs(f[..., 1::2]))

            if is_image:
                f = torch.cat( (f, torch.zeros((f.shape[0:4] + (1,f.shape[5])), device=self.device)), dim=4) # Add the missing channel
            if self.disabled_features is not None:
                f[..., self.disabled_features] = 0

            f_all = torch.cat([
                f[..., 0:4].flatten(start_dim=4),
                f[..., 4:].flatten(start_dim=4)
            ], dim=-1)

            delta = self.transformer_net(f_all) / len(features)

            if bb == len(features)-1:
                delta *= self.baseband_weight
            if is_image:
                delta *= self.image_int

            Q_JOD -= delta

        return Q_JOD

    def full_name(self):
        return "ColorVideoVDP-ML-Transformer"


register_metric( cvvdp_ml_transformer )


class RegressionTransformerBands(nn.Module):
    def __init__(self,
                 in_channels=24,
                 dim=256,
                 depth=4,
                 heads=8,
                 dropout=0.1,
                 max_bands=16):
        super().__init__()
        self.dim = dim
        self.max_bands = max_bands

        self.patch_embed = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.Linear(in_channels, dim),
            Rearrange('b h w c -> b (h w) c')
        )

        self.spatial_pos_mlp = nn.Sequential(
            nn.Linear(2, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim)
        )
        self.band_pos_embed = nn.Embedding(max_bands, dim)
        self.baseband_flag_embed = nn.Embedding(2, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
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

    def get_spatial_position_embedding(self, h, w, device):
        y_coords = (torch.arange(h, device=device).float() + 0.5) / h
        x_coords = (torch.arange(w, device=device).float() + 0.5) / w
        grid = torch.stack(torch.meshgrid(y_coords, x_coords, indexing='ij'), dim=-1)
        pos_embed = self.spatial_pos_mlp(grid)
        return pos_embed.view(1, h * w, self.dim)

    def forward(self, band_features):
        total_bands = len(band_features)
        if total_bands > self.max_bands:
            raise RuntimeError(f"Number of bands ({total_bands}) exceeds max_bands ({self.max_bands})")

        all_band_tokens = []
        batch_sz = band_features[0].shape[0]
        frame_count = band_features[0].shape[1]

        for band_idx, feat in enumerate(band_features):
            B, D, H, W, C = feat.shape
            feat = feat.reshape(B * D, H, W, C).permute(0, 3, 1, 2)

            tokens = self.patch_embed(feat)
            tokens = tokens + self.get_spatial_position_embedding(H, W, feat.device)

            band_index_tensor = torch.full((1,), band_idx, dtype=torch.long, device=feat.device)
            band_pos = self.band_pos_embed(band_index_tensor).view(1, 1, self.dim)
            tokens = tokens + band_pos

            is_baseband = 1 if band_idx == (total_bands - 1) else 0
            baseband_flag_tensor = torch.full((1,), is_baseband, dtype=torch.long, device=feat.device)
            baseband_pos = self.baseband_flag_embed(baseband_flag_tensor).view(1, 1, self.dim)
            tokens = tokens + baseband_pos

            all_band_tokens.append(tokens)

        x = torch.cat(all_band_tokens, dim=1)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)

        cls_feat = x[:, 0]
        y = self.reg_head(cls_feat).squeeze(-1).reshape(batch_sz, frame_count)
        return y.mean(dim=1, keepdim=False)


class cvvdp_ml_transformer_bands(cvvdp_ml):
    def __init__(self,
                 dim=256,
                 config_paths=[],
                 **kwargs):

        self.set_device(kwargs.get('device'))

        self.transformer_net = RegressionTransformerBands(
            in_channels=24,
            dim=dim
        ).to(self.device)

        super().__init__(config_paths=config_paths, **kwargs)

    def get_nets_to_load(self):
        return ['transformer_net']

    def do_pooling_and_jods(self, features):
        batch_sz = features[0].shape[0]
        Q_JOD = torch.ones((batch_sz), device=self.device) * 10.
        is_image = (features[0].shape[4] == 3)

        input_features = []
        for f in features:
            f[..., 1::2] = torch.sqrt(torch.abs(f[..., 1::2]))

            if is_image:
                f = torch.cat((f, torch.zeros((f.shape[0:4] + (1, f.shape[5])), device=self.device)), dim=4)
            if self.disabled_features is not None:
                f[..., self.disabled_features] = 0

            f_all = torch.cat([
                f[..., 0:4].flatten(start_dim=4),
                f[..., 4:].flatten(start_dim=4)
            ], dim=-1)

            input_features.append(f_all)

        delta = self.transformer_net(input_features)
        if is_image:
            delta *= self.image_int
        Q_JOD -= delta

        return Q_JOD

    def full_name(self):
        return "ColorVideoVDP-ML-Transformer-Bands"


register_metric(cvvdp_ml_transformer_bands)

class cvvdp_ml_entropy(cvvdp_ml):

    # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
    # random_init - do not load NN from a checkpoint file, use a random initialization
    def __init__(self, config_paths=[], device=None, **kwargs):

        self.set_device( device )
        super().__init__(config_paths=config_paths, device=device, **kwargs)

    # Perform pooling with per-band weights and map to JODs
    def do_pooling_and_jods(self, features):

        no_bands = len(features)
        batch_sz = features[0].shape[0]

        Q_JOD = torch.ones((batch_sz), device=self.device)*10.

        is_image = (features[0].shape[4]==3) # if 3 channels, it is an image

        for bb in range(no_bands):

            #F[batch,frames,width,height,channels,stat]
            f = features[bb]
            
            if is_image:
                f = torch.cat( (f, torch.zeros((f.shape[0:4] + (1,f.shape[5])), device=self.device)), dim=4) # Add the missing channel
            if self.disabled_features is not None:
                f[..., self.disabled_features] = 0  

            
            entropy = 0.5 * torch.log2(1 + (f[..., 3] + 1e-8) / (f[..., 5] + 1e-8)) 
            f_D = torch.stack( (f[..., 4], entropy), dim=-1 ).flatten( start_dim=4 )

            D_all = self.feature_net(f_D) 
            D_all = F.relu(D_all) /no_bands

            is_base_band = (bb==no_bands-1)
            if is_base_band:
                D_all *= self.baseband_weight

            if is_image:
                D_all *= self.image_int

            Q_JOD -= self.spatiotemporal_pooling(D_all)

        assert(not Q_JOD.isnan().any())
        return Q_JOD

    def full_name(self):
        return "ColorVideoVDP-ML-Entropy"

    def spatiotemporal_pooling(self, D_all):
        return D_all.view(D_all.shape[0],-1).mean(dim=1)

register_metric( cvvdp_ml_entropy )

class cvvdp_ml_snr(cvvdp_ml_base):

    # use_checkpoints - this is for memory-efficient gradient propagation (to be used with stage1 training only)
    # random_init - do not load NN from a checkpoint file, use a random initialization
    def __init__(self, config_paths=[], device=None, **kwargs):

        self.set_device( device )

        dropout = 0.2
        hidden_dims = 48
        num_layers = 4
        ch_no = 4 # 4 visual channels: A_sust, A_trans, RG, YV
        stats_no = 4 # mean D, var D, snr, entropy
        self.feature_net = MLP(in_channels=stats_no*ch_no, hidden_channels=[hidden_dims]*num_layers + [1], activation_layer=torch.nn.ReLU, dropout=dropout).to(self.device)

        super().__init__(config_paths=config_paths, device=device, **kwargs)
    
    def get_nets_to_load(self):
        return [ 'feature_net' ]

    # Perform pooling with per-band weights and map to JODs
    def do_pooling_and_jods(self, features):

        no_bands = len(features)
        batch_sz = features[0].shape[0]

        Q_JOD = torch.ones((batch_sz), device=self.device)*10.

        is_image = (features[0].shape[4]==3) # if 3 channels, it is an image

        for bb in range(no_bands):

            #F[batch,frames,width,height,channels,stat]
            f = features[bb]
            
            if is_image:
                f = torch.cat( (f, torch.zeros((f.shape[0:4] + (1,f.shape[5])), device=self.device)), dim=4) # Add the missing channel
            if self.disabled_features is not None:
                f[..., self.disabled_features] = 0  

            
            entropy = 0.5 * torch.log2(1 + (f[..., 3] + 1e-8) / (f[..., 5] + 1e-8)) 
            snr = 10 * torch.log10( 1 + (f[..., 4]**2 + 1e-8) / (f[..., 2]**2 + 1e-8) )
            f_D = torch.stack( (f[..., 4], f[..., 5], snr, entropy), dim=-1 ).flatten( start_dim=4 )

            D_all = self.feature_net(f_D) 
            D_all = F.relu(D_all) /no_bands

            is_base_band = (bb==no_bands-1)
            if is_base_band:
                D_all *= self.baseband_weight

            if is_image:
                D_all *= self.image_int

            Q_JOD -= self.spatiotemporal_pooling(D_all)

        assert(not Q_JOD.isnan().any())
        return Q_JOD

    def full_name(self):
        return "ColorVideoVDP-ML-SNR"

    def spatiotemporal_pooling(self, D_all):
        return D_all.view(D_all.shape[0],-1).mean(dim=1)

register_metric( cvvdp_ml_snr )

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
