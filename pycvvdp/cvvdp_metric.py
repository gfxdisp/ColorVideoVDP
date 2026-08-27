from abc import abstractmethod
from urllib.parse import ParseResultBytes
try:
    from numpy import expand_dims
except ImportError:
    from numpy.lib.shape_base import expand_dims
import math
from dataclasses import dataclass, field
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
from tqdm import tqdm
from datetime import date

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
from pycvvdp.video_writer import VideoWriter, ImageWriter

#from pycvvdp.colorspace import lms2006_to_dkld65

# For debugging only
# from gfxdisp.pfs.pfs_torch import pfs_torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from third_party.cpuinfo import cpuinfo
from pycvvdp.lpyr_dec import lpyr_dec, lpyr_dec_2, weber_contrast_pyr, log_contrast_pyr
from interp import interp1, interp3, interp1dim2

import pycvvdp.utils as utils

from pycvvdp.display_model import vvdp_display_photometry, vvdp_display_geometry
from pycvvdp.csf import castleCSF

# import gc
# def print_large_tensors():
#     print( '---------------' )
#     objs = []
#     for obj in gc.get_objects():
#         try:
#             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                 if len(obj.size()) > 0:
#                     mem_used = obj.nelement() * obj.element_size()
#                     if mem_used > 1000:
#                         objs.append( (mem_used, f"{mem_used/1000000000:.2f} GB {type(obj)} {obj.size()}") )
#         except: # (KeyError, AttributeError, RuntimeError):
#             pass
#     objs_sorted = sorted( objs, key=lambda obj: obj[0], reverse=True )
#     for obj in objs_sorted:
#         print( obj[1] )

class SeparableGaussianBlur:
    """
    Compute two sequential orthogonal 1D convolutions.
    Faster than performing one 2D convolution.
    """

    def __init__(self, kernel_size, sigma, device=None, dtype=torch.float32):
        k = int(kernel_size)
        half = (k - 1) * 0.5
        x = torch.linspace(-half, half, k, device=device, dtype=dtype)
        pdf = torch.exp(-0.5 * (x / float(sigma)) ** 2)
        k1d = pdf / pdf.sum()
        self.pad = (k // 2,) * 4  # (left, right, top, bottom)
        self.k_vert = k1d.view(1, 1, k, 1)
        self.k_horiz = k1d.view(1, 1, 1, k)

    def to(self, device):
        self.k_vert = self.k_vert.to(device)
        self.k_horiz = self.k_horiz.to(device)
        return self

    def forward(self, img):  # img: (N, 1, H, W)
        x = Func.pad(img, self.pad, mode="reflect")
        x = Func.conv2d(x, self.k_vert)
        x = Func.conv2d(x, self.k_horiz)
        return x

    __call__ = forward



# A differentiable variant of a power function
def safe_pow( x:Tensor, p ): 
    #assert (not x.isnan().any()) and (not x.isinf().any()), "Must not be nan"
    #assert torch.all(x>=0), "Must be positive"

    if True: #isinstance( p, Tensor ) and p.requires_grad:
        # If we need a derivative with respect to p, x must not be 0
        epsilon = torch.as_tensor( 0.00001, device=x.device )
        return (x+epsilon) ** p - epsilon**p
    else:
        return x ** p


# A power function that can handle negative values (by preserving the sign)
def pow_neg( x:Tensor, p ): 
    #assert (not x.isnan().any()) and (not x.isinf().any()), "Must not be nan"

    #return torch.tanh(100*x) * (torch.abs(x) ** p)

    min_v = torch.as_tensor( 0.00001, device=x.device )
    return (torch.max(x,min_v) ** p) + (torch.max(-x,min_v) ** p) - min_v**p


class cvvdp_frame_buffers:
    def __init__(self) -> None:
        self.sw_buf = [None, None] # Sliding window buffer [test: Tensor, reference: Tensor] - stores frames for applying a temporal filter
        self.ra_buf = [[], []] # Read-ahead buffer [test: List, reference: List] - used for the symmetric padding


@dataclass
class RefCache:
    """
    Precomputed, reference-only intermediates for cvvdp.*_with_cached_reference(),
    valid only when masking_model=="mult-ref" and contrast is a *_ref mode (where
    S_test==S_ref and both, plus the masking normalizer M, depend only on the
    reference image - see precompute_reference()). Images only (no video).
    """
    bands: list             # per band: {'S_ref': ...} (baseband) or {'S_ref','R_p','M'} (masking bands)
    reference: torch.Tensor # the raw reference tensor, kept so it can be re-interleaved with a new test tensor each call
    width: int
    height: int
    masking_model: str
    contrast: str
    dim_order: str
    frames_per_second: float


"""
ColorVideoVDP metric. Refer to pytorch_examples for examples on how to use this class. 

spatial_padding - how to pad areas outside the image when computing a multi-scale decomposition. 
    Pass: "zero" - zero padding, "symmetric" - mirror-like reflection, "valid" - ignore edges, None - use model's default (recommended)

temp_padding - how to pad frames before the first frame in the sequence when computing temporal filters. 
    Pass: "replicate" - repeat the first frame, "symmetric" - mirror-like reflection, None - use model's default (recommended)
"""
class cvvdp(vq_metric):
    def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, config_paths=[], 
                 heatmap=None, quiet=False, device=None, temp_padding=None, spatial_padding=None, use_checkpoints=False, 
                 dump_channels=None, gpu_mem = None):
        self.quiet = quiet
        self.heatmap = heatmap
        self.heatmap_file = None
        self.heatmap_vw = None
        self.temp_padding = temp_padding
        self.spatial_padding = spatial_padding
        self.use_checkpoints = use_checkpoints # Used for end-to-end training, these are NOT model checkpoints
        self.gpu_mem = gpu_mem # how many GB of memory we are allowed to use
        self.training_mode = False

        assert heatmap in ["threshold", "supra-threshold", "raw", "none", None], "Unknown heatmap type"            

        self.do_heatmap = (not self.heatmap is None) and (self.heatmap != "none")

        # Use GPU if available
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count()>0:
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        self.set_display_model(display_name, display_photometry=display_photometry, display_geometry=display_geometry, config_paths=config_paths)

        self.temp_resample = False  # When True, resample the temporal features to nominal_fps
        self.nominal_fps = 240

        self.load_config(config_paths)
        # if calibrated_ckpt is not None:
        #     self.update_from_checkpoint(calibrated_ckpt)

        self.dump_channels = dump_channels
        self.heatmap_pyr = None

    # Switch to training mode (e.g., to optimize memory allocation)
    def train(self, do_training=True):
        self.training_mode = do_training

    def load_config( self, config_paths ):

        #parameters_file = os.path.join(os.path.dirname(__file__), "fvvdp_data/fvvdp_parameters.json")
        self.parameters_file = utils.config_files.find( "cvvdp_parameters.json", config_paths )
        logging.debug( f"Loading ColorVideoVDP parameters from '{self.parameters_file}'" )
        parameters = utils.json2dict(self.parameters_file)

        #all common parameters between Matlab and Pytorch, loaded from the .json file
        self.mask_p = torch.as_tensor( parameters['mask_p'], device=self.device )
        self.mask_c = torch.as_tensor( parameters['mask_c'], device=self.device ) # content masking adjustment
        self.pu_dilate = parameters['pu_dilate']
        if self.pu_dilate>0:
            self.pu_blur = SeparableGaussianBlur(int(self.pu_dilate*4)+1, self.pu_dilate, device=self.device)
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
        if self.spatial_padding is None:
            self.spatial_padding = parameters.get( 'spatial_padding', 'symmetric' )
        if self.temp_padding is None:
            self.temp_padding = parameters.get( 'temp_padding', 'symmetric' )

        if 'ce_g' in parameters:
            self.ce_g = torch.as_tensor( parameters['ce_g'], device=self.device )

        if 'k_c' in parameters:
            self.k_c = torch.as_tensor( parameters['k_c'], device=self.device )

        if 'temp_filter' in parameters:
            self.temp_filter = parameters['temp_filter']
        else:
            self.temp_filter = "default"

        if 'mask_q' in parameters:
            self.mask_q = torch.as_tensor( parameters['mask_q'], device=self.device )
        else:
            self.mask_q_sust = torch.as_tensor( parameters['mask_q_sust'], device=self.device )
            self.mask_q_trans = torch.as_tensor( parameters['mask_q_trans'], device=self.device )
        self.filter_len = torch.as_tensor( parameters['filter_len'], device=self.device )

        self.do_xchannel_masking = True if parameters['xchannel_masking'] == "on" else False
        self.xcm_weights = torch.as_tensor( parameters['xcm_weights'], device=self.device, dtype=torch.float32 )
        # Precompute the cross-channel masking weights (constant per model); consumed by mask_pool's einsum.
        self.xcm_pow = 2**self.xcm_weights

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
        # if self.baseband_weight.numel()<4:
        #     self.baseband_weight = self.baseband_weight.repeat(4)
        self.dclamp_type = parameters['dclamp_type']  # clamping mode: soft or hard
        self.d_max = torch.as_tensor( parameters['d_max'], device=self.device ) # Clamping of difference values
        # Precompute elementwise powers
        self.mask_c_pow = 10.0**self.mask_c
        self.d_max_pow = 10.0**self.d_max
        self.version = parameters['version']

        self.lum_adapt_reference = parameters.get( 'lum_adapt_reference', True )
        self.omega = [0, 5]

        # If True, the baseband's CSF spatial frequency is derived from the actual
        # difference content (differentiable spectral centroid) instead of a fixed
        # 0.1 cpd, and baseband_weight is not applied. Off by default for full
        # backward compatibility.
        self.baseband_freq_adapt = parameters.get( 'baseband_freq_adapt', False )

        # If True, an extra whole-frame "DC" term is appended after the last pyramid
        # band: a Weber-law-normalized, full-resolution (pre-pyramid) mean opponent-
        # channel difference. It exists to catch large-area/near-uniform casts that the
        # CSF-frequency-based baseband term structurally can't see (their spectral
        # centroid collapses to the CSF floor, same as a real DC pedestal, and the
        # pyramid crushes the whole frame down to a tiny baseband grid before the
        # spatial-frequency machinery ever runs). Off by default for full backward
        # compatibility; when on, its per-channel weight is `dc_weight`, calibrated the
        # same way as `baseband_weight` (a free multiplier fit in do_pooling_and_jods).
        self.dc_term_enabled = parameters.get( 'dc_term_enabled', False )
        self.dc_weight = torch.as_tensor( parameters.get( 'dc_weight', [1.0, 1.0, 1.0, 1.0] ), device=self.device )

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
            if hasattr(display_photometry, 'short_name'):
                self.display_name = display_photometry.short_name
            else:
                self.display_name = "unspecified"
        
        if display_geometry is None:
            self.display_geometry = vvdp_display_geometry.load(display_name, config_paths)
        else:
            self.display_geometry = display_geometry

        self.pix_per_deg = self.display_geometry.get_ppd()
        #self.imgaussfilt = utils.ImGaussFilt(0.5 * self.pix_per_deg, self.device)
        self.lpyr = None

    '''
    Predict image/video quality using ColorVideoVDP.

    test_cont and reference_cont can be either numpy arrays or PyTorch tensors with images or video frames. 
        Depending on the display model (display_photometry), the pixel values should be either display encoded, or absolute linear.
        The two supported datatypes are float16 and uint8.
    dim_order - a string with the order of dimensions of test_cont and reference_cont. The individual characters denote
        B - batch
        C - color channel
        F - frame
        H - height
        W - width
        Examples: "HW" - gray-scale image (column-major pixel order); "HWC" - color image; "FCHW" - color video
        The default order is "BCFHW". The processing can be a bit faster if data is provided in that order. 
    frame_padding - the metric requires at least 250ms of video for temporal processing. Because no previous frames exist in the
        first 250ms of video, the metric must pad those first frames. This options specifies the type of padding to use:
          'replicate' - replicate the first frame (default)
          'symmetric'  - the video frames are mirrored so that frames -1, -2, ... correspond to frames 0, 1, ...
    '''
    def predict(self, test_cont, reference_cont, dim_order="BCFHW", frames_per_second=0, heatmap_file=None):

        test_vs = video_source_array( test_cont, reference_cont, frames_per_second, dim_order=dim_order, display_photometry=self.display_photometry )

        return self.predict_video_source(test_vs, heatmap_file=heatmap_file)

    '''
    Compute a loss function between test and reference images/videos. Used as an optimization term in which the loss is minimized. 
    '''
    def loss(self, test_cont, reference_cont, dim_order="BCFHW", frames_per_second=0):

        test_vs = video_source_array( test_cont, reference_cont, frames_per_second, dim_order=dim_order, display_photometry=self.display_photometry )
        (Q_jod, stats) = self.predict_video_source(test_vs)
        return (10.-Q_jod)


    '''
    The same as `predict` but takes as input fvvdp_video_source_* object instead of Numpy/Pytorch arrays. Video source is recommended when processing long videos as it allows frame-by-frame loading.
    '''
    def predict_video_source(self, vid_source, heatmap_file=None ):
        # We assume the pytorch default NCDHW layout

        self.heatmap_file = heatmap_file

        vid_sz = vid_source.get_video_size() # H, W, F
        height, width, N_frames = vid_sz
        batch_sz = vid_source.get_batch_size()

        if batch_sz>1 and (self.heatmap is not None and self.heatmap!='none'):
            raise vq_exception( 'Heatmaps not supported when batches are used' )
        
        assert batch_sz==1 or self.device.type != 'mps', "Batch mode curretly does not work correctly with MPS (most likely due to a PyTorch bug). Run on a CPU."

        # 'medium' is a bit slower than 'high' on 3090
        # torch.set_float32_matmul_precision('medium')

        if self.lpyr is None or self.lpyr.W!=width or self.lpyr.H!=height:
            if self.contrast.startswith("weber"):
                self.lpyr = weber_contrast_pyr(width, height, self.pix_per_deg, self.device, contrast=self.contrast, padding_type=self.spatial_padding)
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

        no_channels = 2+temp_ch

        if self.do_heatmap and self.heatmap_file is None:
            dmap_channels = 1 if self.heatmap == "raw" else 3
            heatmap = torch.zeros([1,dmap_channels,N_frames,height,width], dtype=torch.float16, device=torch.device('cpu')) # Store heatmap in the CPU memory
        else:
            heatmap = None

        Q_per_ch = None

        if self.device.type == 'cuda' and torch.cuda.is_available() and not is_image:
            # GPU utilization is better if we process many frames, but it requires more GPU memory
            pix_cnt = width*height*batch_sz
            block_N_frames = max(1, min(self.estimate_block_N(pix_cnt, self.filter_len), N_frames))
        else:
            block_N_frames = 1

        if self.debug:
            logging.debug( f"Processing a block of {block_N_frames} frames at a time.")

        # block_N_frames = min(block_N_frames,1)
        # print( f'Block of frames: {block_N_frames}')

        if self.contrast=="log":
            met_colorspace='logLMS_DKLd65'
        else:
            met_colorspace='DKLd65' # This metric uses DKL colorspace with d65 whitepoint

        if self.dump_channels:
            self.dump_channels.open(vid_source.get_frames_per_second())

        show_progress_bar = not is_image and not self.quiet

        fb = cvvdp_frame_buffers()
        # sw_buf = [None, None] # Sliding window buffer [test, reference] - stores frames for applying a temporal filter
        # ra_buf = [None, None] # Read-ahead buffer [test, reference]

        for ff in tqdm(range(0, N_frames, block_N_frames), disable=not show_progress_bar):
            cur_block_N_frames = min(block_N_frames,N_frames-ff) # How many frames in this block?

            R = self.read_block_of_frames(vid_source, no_channels, fb, block_N_frames, met_colorspace, ff, cur_block_N_frames)

            if self.dump_channels:
                self.dump_channels.dump_temp_ch(R)

            if self.use_checkpoints:
                # Used for training
                Q_per_ch_block, heatmap_block = checkpoint.checkpoint(self.process_block_of_frames, R, vid_sz, temp_ch, self.lpyr, is_image, use_reentrant=False)
            else:
                Q_per_ch_block, heatmap_block = self.process_block_of_frames(R, vid_sz, temp_ch, self.lpyr, is_image)

            if Q_per_ch is None:
                Q_per_ch = torch.zeros((batch_sz,Q_per_ch_block.shape[1], N_frames, Q_per_ch_block.shape[3]), device=self.device)
            
            ff_end = ff+Q_per_ch_block.shape[2]
            Q_per_ch[:,:,ff:ff_end,:] = Q_per_ch_block  

            # print_large_tensors()

            if self.do_heatmap:
                if self.heatmap == "raw":
                    heatmap_vis = heatmap_block[0,...].detach()
                else:
                    ref_frame = R[:,0, :, :, :]
                    heatmap_vis = visualize_diff_map(heatmap_block, context_image=ref_frame, colormap_type=self.heatmap, use_cpu=self.device.type == 'mps').detach()

                if self.heatmap_file is None:
                    heatmap[:,:,ff:ff_end,...] = heatmap_vis.type(torch.float16).cpu()
                else:
                    if self.heatmap_vw is None:
                        if is_image:
                            self.heatmap_vw = ImageWriter( self.heatmap_file )
                        else:
                            self.heatmap_vw = VideoWriter( self.heatmap_file, hdr_mode=False, fps=vid_source.get_frames_per_second() )
    
                    for kk in range(heatmap_vis.shape[1]):
                        if heatmap_vis.shape[0]==1: # grayscale heatmap
                            self.heatmap_vw.write_frame_rgb((heatmap_vis[:,kk,:,:]/10).view((height,width,1)).tile((1,1,3)).cpu().numpy())
                        else:
                            self.heatmap_vw.write_frame_rgb(heatmap_vis[:,kk,:,:].permute((1,2,0)).cpu().numpy())


        if self.temp_resample: # This may not be needed anymore
            t_end = N_frames/vid_source.get_frames_per_second() # Video duration in s
            t_org = torch.linspace( 0., t_end, N_frames, device=self.device )
            N_frames_resampled = math.ceil(t_end * self.nominal_fps)
            t_resampled = torch.linspace( 0., N_frames_resampled/self.nominal_fps, N_frames_resampled, device=self.device )
            Q_per_ch = interp1dim2(t_org, Q_per_ch, t_resampled)
            N_frames = N_frames_resampled
            fps = self.nominal_fps
        else:
            fps = vid_source.get_frames_per_second()

        # print( Q_per_ch.mean(dim=(0,2)) )
        rho_band = self.lpyr.get_freqs()
        Q_jod = self.do_pooling_and_jods(Q_per_ch)

        stats = {}
        stats['Q_per_ch'] = Q_per_ch.detach().cpu().numpy() # the quality per channel and per frame
        stats['rho_band'] = rho_band # The spatial frequency per band in cpd
        stats['frames_per_second'] = fps
        stats['width'] = width
        stats['height'] = height
        stats['N_frames'] = N_frames

        if self.dump_channels:
            self.dump_channels.close()

        if self.heatmap_vw is not None:
            self.heatmap_vw.close()

        if heatmap is not None:
            stats['heatmap'] = heatmap

        if self.debug: 
            logging.debug( f"Processing {block_N_frames} frames in a batch." )
            logging.debug( f"Filter length {self.filter_len} frames." )
            logging.debug( f"Resolution: {width}x{height} = {width*height/1e6} Mpixels" )
            # mem_allocated_peak = torch.cuda.max_memory_allocated(self.device)            
            # logging.debug( f"Memory allocated at start: {self.start_allocated/1e9} GB" )
            if hasattr( self, "sw_buf_allocated" ):
                logging.debug( f"Memory allocated for temp. filter buffers: {self.sw_buf_allocated/1e9} GB" )
            logging.debug( f"Max memory allocated: {torch.cuda.max_memory_allocated()/1e9} GB" )
            # The model is:  total_mem = a + pix_cnt*(block_N_frames+filter_len-1)*b + pix_cnt*block_N_frames*c



        return (Q_jod.squeeze(), stats)

    def _ensure_lpyr_for_size(self, width, height):
        if self.lpyr is None or self.lpyr.W!=width or self.lpyr.H!=height:
            if self.contrast.startswith("weber"):
                self.lpyr = weber_contrast_pyr(width, height, self.pix_per_deg, self.device, contrast=self.contrast, padding_type=self.spatial_padding)
            elif self.contrast.startswith("log"):
                self.lpyr = log_contrast_pyr(width, height, self.pix_per_deg, self.device, contrast=self.contrast)
            else:
                raise RuntimeError( f"Unknown contrast {self.contrast}" )

    '''
    Precompute reference-only intermediates (S_ref, and per masking band R_p, M)
    for a fixed reference image, to be reused across many calls to
    predict_with_cached_reference()/loss_with_cached_reference() against
    different test images - e.g. when cvvdp is used as a loss function to
    optimize a test image toward a fixed reference (image recovery, style
    transfer, etc). See RefCache for what's stored and why it's valid to cache.

    Requires masking_model=="mult-ref" and contrast in ("weber_g1_ref",
    "weber_g0_ref") - the only combination where S_test==S_ref and the masking
    normalizer M depend solely on the reference. Images only (no video).
    '''
    def precompute_reference(self, reference, dim_order="CHW", frames_per_second=0):
        if self.masking_model != "mult-ref":
            raise RuntimeError( f"precompute_reference() requires masking_model=='mult-ref', got '{self.masking_model}'" )
        if self.contrast not in ("weber_g1_ref", "weber_g0_ref"):
            raise RuntimeError( f"precompute_reference() requires a *_ref contrast mode, got '{self.contrast}'" )
        if self.dc_term_enabled or self.baseband_freq_adapt:
            raise RuntimeError( "precompute_reference() does not support dc_term_enabled or baseband_freq_adapt - both still depend on the live test image every call, so the cache would be silently wrong." )

        ref_vs = video_source_array( reference, reference, frames_per_second, dim_order=dim_order, display_photometry=self.display_photometry )
        height, width, N_frames = ref_vs.get_video_size()
        if N_frames != 1:
            raise NotImplementedError( "precompute_reference() currently supports images only (N_frames==1)." )

        self._ensure_lpyr_for_size(width, height)

        met_colorspace = 'logLMS_DKLd65' if self.contrast=="log" else 'DKLd65'
        batch_sz = ref_vs.get_batch_size()
        ref_frame = ref_vs.get_reference_frame(0, device=self.device, colorspace=met_colorspace)
        R = torch.empty((batch_sz, 6, 1, height, width), device=self.device)
        R[:,0::2,:,:,:] = ref_frame
        R[:,1::2,:,:,:] = ref_frame

        capture_ref = []
        with torch.no_grad():
            self.process_block_of_frames(R, (height,width,1), 1, self.lpyr, True, capture_ref=capture_ref)

        return RefCache(bands=capture_ref, reference=ref_frame.detach(), width=width, height=height,
                         masking_model=self.masking_model, contrast=self.contrast,
                         dim_order=dim_order, frames_per_second=frames_per_second)

    '''
    Fast-path counterpart to predict(), reusing a RefCache from
    precompute_reference() instead of recomputing reference-only quantities.
    Returns Q_jod, with the same convention as predict()/predict_video_source().
    '''
    def predict_with_cached_reference(self, test, ref_cache: 'RefCache', dim_order="CHW"):
        if self.masking_model != ref_cache.masking_model or self.contrast != ref_cache.contrast:
            raise RuntimeError( "Metric configuration (masking_model/contrast) differs from what precompute_reference() was called with - the cache is no longer valid for this metric instance." )

        test_vs = video_source_array( test, test, ref_cache.frames_per_second, dim_order=dim_order, display_photometry=self.display_photometry )
        height, width, N_frames = test_vs.get_video_size()
        if N_frames != 1:
            raise NotImplementedError( "predict_with_cached_reference() currently supports images only (N_frames==1)." )
        if width != ref_cache.width or height != ref_cache.height:
            raise RuntimeError( f"Test image size ({width}x{height}) does not match the size RefCache was built for ({ref_cache.width}x{ref_cache.height})." )

        self._ensure_lpyr_for_size(width, height)

        met_colorspace = 'logLMS_DKLd65' if self.contrast=="log" else 'DKLd65'
        batch_sz = test_vs.get_batch_size()
        R = torch.empty((batch_sz, 6, 1, height, width), device=self.device)
        R[:,0::2,:,:,:] = test_vs.get_test_frame(0, device=self.device, colorspace=met_colorspace)
        R[:,1::2,:,:,:] = ref_cache.reference

        Q_per_ch_block, _ = self.process_block_of_frames(R, (height,width,1), 1, self.lpyr, True, ref_cache=ref_cache)
        Q_jod = self.do_pooling_and_jods(Q_per_ch_block)
        return Q_jod.squeeze()

    '''
    Same as predict_with_cached_reference() but returns a loss (10-JOD), with
    the same convention as loss().
    '''
    def loss_with_cached_reference(self, test, ref_cache: 'RefCache', dim_order="CHW"):
        return 10. - self.predict_with_cached_reference(test, ref_cache, dim_order=dim_order)

    # Get a positive index of a frame for symmetric padding
    # If a video is too short to match the filter length, in will replicate frames back and forth in a ping-pong manner
    def _get_symmetric_frame_index( self, frame_ind, frame_count ):
        is_even = (math.floor((abs(frame_ind)-1)/(frame_count-1)) % 2)==0
        if is_even:
            return ((abs(frame_ind)-1) % (frame_count-1))+1
        else:
            return (frame_ind % (frame_count-1))

    # Reads a block of frames and applies temporal filter, as needed
    def read_block_of_frames(self, vid_source, no_channels, fb, block_N_frames, met_colorspace, ff, cur_block_N_frames):

        vid_sz = vid_source.get_video_size() # H, W, F
        height, width, N_frames = vid_sz
        batch_sz = vid_source.get_batch_size()
        fl = self.filter_len

        is_image = (N_frames==1)  # Can run faster on images

        if is_image:
            R = torch.empty((batch_sz, 6, 1, height, width), device=self.device)
            R[:,0::2, :, :, :] = vid_source.get_test_frame(0, device=self.device, colorspace=met_colorspace)
            R[:,1::2, :, :, :] = vid_source.get_reference_frame(0, device=self.device, colorspace=met_colorspace)

        else: # This is video
            #if self.debug: print("Frame %d:\n----" % ff)
            # Check if video source provides pre-filtered temporal channels (e.g., SPEM extension)
            is_pre_filtered = getattr(vid_source, 'is_temporally_filtered', False)

            if is_pre_filtered:
                # Bypass path: frames are already temporally filtered
                # Output R structure: [T-AchS, R-AchS, T-RG, R-RG, T-YV, R-YV, T-AchT, R-AchT]
                R = torch.zeros((batch_sz, 8, cur_block_N_frames, height, width), device=self.device)

                for fi in range(cur_block_N_frames):
                    # Fetch reference first so optical flow is computed and cached
                    # before test frame needs it (SPEM computes flow from reference only)
                    R_filt = vid_source.get_reference_frame(ff+fi, device=self.device, colorspace='DKLd65_trans')
                    T_filt = vid_source.get_test_frame(ff+fi, device=self.device, colorspace='DKLd65_trans')

                    # Map to interleaved 8-channel structure
                    # Channels 0-2: DKL sustained (Ach, RG, YV)
                    # Channel 3: DKL Ach-Transient
                    for ch in range(4):
                        R[:, ch*2, fi, :, :] = T_filt[:, ch, 0, :, :]
                        R[:, ch*2+1, fi, :, :] = R_filt[:, ch, 0, :, :]

            else:
                # Standard path: perform temporal filtering
                if ff == 0: # First frame
                    buf_len = fl+block_N_frames-1
                    fb.sw_buf[0] = torch.zeros((batch_sz,3,buf_len,height,width), device=self.device, dtype=torch.float32)  # In some cases, it should be possible to use float16
                    fb.sw_buf[1] = torch.zeros((batch_sz,3,buf_len,height,width), device=self.device, dtype=torch.float32)

                    if self.debug and not hasattr( self, 'sw_buf_allocated' ):
                            # Memory allocated after creating buffers for temporal filters
                        self.sw_buf_allocated = torch.cuda.max_memory_allocated(self.device)

                    for fi in range(cur_block_N_frames):
                        ind = fl+fi-1
                        fb.sw_buf[0][:,:,ind:ind+1,:,:] = vid_source.get_test_frame(ff+fi, device=self.device, colorspace=met_colorspace)
                        fb.sw_buf[1][:,:,ind:ind+1,:,:] = vid_source.get_reference_frame(ff+fi, device=self.device, colorspace=met_colorspace)

                    if self.temp_padding == "replicate":
                        ind = fl-1
                        fb.sw_buf[0][:,:,0:fl-1,:,:] = fb.sw_buf[0][:,:,ind:ind+1,:,:] # Replicate the first frame
                        fb.sw_buf[1][:,:,0:fl-1,:,:] = fb.sw_buf[1][:,:,ind:ind+1,:,:] # Replicate the first frame

                    elif self.temp_padding == "symmetric":

                        # If the cur_block_N_frames smaller than the filter length, we need to read those frames ahead
                        frames_in_video = N_frames-ff-cur_block_N_frames # How many frames are left unread in the video
                        for fi in range(min(max(fl-cur_block_N_frames,0),frames_in_video)):
                            ind = ff+cur_block_N_frames+fi
                            fb.ra_buf[0].append(vid_source.get_test_frame(ind, device=self.device, colorspace=met_colorspace))
                            fb.ra_buf[1].append(vid_source.get_reference_frame(ind, device=self.device, colorspace=met_colorspace))

                        for fi in range(-fl+1,0):
                            pos_ind = self._get_symmetric_frame_index(fi, N_frames)
                            buf_ind = fi+fl-1
                            if pos_ind < cur_block_N_frames: # whether the frame is in current buffer
                                sw_ind = pos_ind + fl-1
                                fb.sw_buf[0][:,:,buf_ind:buf_ind+1,:,:] = fb.sw_buf[0][:,:,sw_ind:sw_ind+1,:,:]
                                fb.sw_buf[1][:,:,buf_ind:buf_ind+1,:,:] = fb.sw_buf[1][:,:,sw_ind:sw_ind+1,:,:]
                            else: # use the read-ahead buffer
                                ra_ind = pos_ind - cur_block_N_frames
                                fb.sw_buf[0][:,:,buf_ind:buf_ind+1,:,:] = fb.ra_buf[0][ra_ind]
                                fb.sw_buf[1][:,:,buf_ind:buf_ind+1,:,:] = fb.ra_buf[1][ra_ind]

                    else:
                        raise RuntimeError( f'Unknown padding method "{self.temp_padding}"' )
                else:
                    # scroll the sliding window buffers
                    # Tensor splicing leads to strange errors with videos; switching to torch.roll()
                    # sw_buf[0][:,:,0:-cur_block_N_frames,:,:] = sw_buf[0][:,:,cur_block_N_frames:,:,:]
                    # sw_buf[1][:,:,0:-cur_block_N_frames,:,:] = sw_buf[1][:,:,cur_block_N_frames:,:,:]
                    fb.sw_buf[0] = torch.roll(fb.sw_buf[0], shifts=-block_N_frames, dims=2)
                    fb.sw_buf[1] = torch.roll(fb.sw_buf[1], shifts=-block_N_frames, dims=2)

                    for fi in range(cur_block_N_frames):
                        ind=fl+fi-1
                        if fb.ra_buf[0]: # if read-ahead buffer is not empty
                            fb.sw_buf[0][:,:,ind:ind+1,:,:] = fb.ra_buf[0].pop(0)
                            fb.sw_buf[1][:,:,ind:ind+1,:,:] = fb.ra_buf[1].pop(0)
                        else:
                            fb.sw_buf[0][:,:,ind:ind+1,:,:] = vid_source.get_test_frame(ff+fi, device=self.device, colorspace=met_colorspace)
                            fb.sw_buf[1][:,:,ind:ind+1,:,:] = vid_source.get_reference_frame(ff+fi, device=self.device, colorspace=met_colorspace)

                # Channel order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y
                # Images do not have the two last channels
                R = torch.zeros((batch_sz, 8, cur_block_N_frames, height, width), device=self.device)

                for cc in range(no_channels): # Iterate over chromatic and temporal channels
                        # 1D filter over time (over frames)
                    corr_filter = self.F[cc].flip(0).view([1,1,self.F[cc].shape[0],1,1])
                    sw_ch = 0 if cc==3 else cc # color channel in the sliding window
                    for fi in range(cur_block_N_frames):
                        R[:,(cc*2+0):(cc*2+1), fi:(fi+1), :, :] = (fb.sw_buf[0][:, sw_ch:(sw_ch+1), fi:(fl+fi), :, :] * corr_filter).sum(dim=-3,keepdim=True) # Test
                        R[:,(cc*2+1):(cc*2+2), fi:(fi+1), :, :] = (fb.sw_buf[1][:, sw_ch:(sw_ch+1), fi:(fl+fi), :, :] * corr_filter).sum(dim=-3,keepdim=True) # Reference
        return R

    # Determine how many frames we can process in a single batch 
    # Larger batch means faster processing, but it requires more memory
    def estimate_block_N(self, pix_cnt, filter_len):
        # Determine how much memory we have
        free, _ = torch.cuda.mem_get_info()
        cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        mem_avail_pytorch = free + cached 

        if has_nvml:
            # This is a more accurate estimate
            nvmlInit()
            dev_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            h = nvmlDeviceGetHandleByIndex(dev_index)
            info = nvmlDeviceGetMemoryInfo(h)
            cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            mem_avail_nvml = info.free + cached  
            mem_avail = min( mem_avail_nvml, mem_avail_pytorch)
        else:
            mem_avail = mem_avail_pytorch
            mem_avail_nvml = None

        if not self.gpu_mem is None:
            mem_avail = min(int(self.gpu_mem*1e9), mem_avail)

        # Estimate how much we need for processing
        # The model is:  total_mem = a + pix_cnt*(block_N_frames+filter_len-1)*b + pix_cnt*block_N_frames*c
        a = 1.6e9
        b = 12 # 3 channel x 4-byte float
        if os.name == 'nt': # For some reason, more memory is consumed on Windows
            c = 450 if not self.training_mode else 800 # A different value for training
        else:
            c = 400 if not self.training_mode else 800 # A different value for training


        block_N_frames = int(math.floor((mem_avail-a-pix_cnt*(filter_len-1)*b)/(pix_cnt*b+pix_cnt*c))) # how many frames can we fit into memory

        if self.debug:
            logging.debug( f"Available memory (PyTorch): {mem_avail_pytorch/1e9} GB")            
            logging.debug( f"Available memory (NVML): {mem_avail_nvml/1e9} GB")
            total_mem_est = a + pix_cnt*(block_N_frames+filter_len-1)*b + pix_cnt*block_N_frames*c
            logging.debug( f"Estimated memory use: {total_mem_est/1e9} GB")

        return block_N_frames


    def get_ch_weights(self, no_channels):
        if hasattr(self, 'ch_chrom_w'):
            per_ch_w_all = torch.stack( [torch.as_tensor(1., device=self.ch_chrom_w.device), self.ch_chrom_w, self.ch_chrom_w, self.ch_trans_w] )
        else:
            # Depreciated - will be removed later
            per_ch_w_all = self.ch_weights
            
        # Weights for the channels: sustained, RG, YV, [transient]
        per_ch_w = per_ch_w_all[0:no_channels].view(1,-1,1,1)
        return per_ch_w


    # Perform pooling with per-band weights and map to JODs
    def do_pooling_and_jods(self, Q_per_ch):
        # Q_per_ch[batch,channel,frame,sp_band]

        no_channels = Q_per_ch.shape[1]
        no_frames = Q_per_ch.shape[2]
        no_bands = Q_per_ch.shape[3]

        per_ch_w = self.get_ch_weights( no_channels )

        # Weights for the spatial bands
        per_sband_w = torch.ones( (1,no_channels,1,no_bands), dtype=torch.float32, device=self.device)
        # When the DC term is enabled it's appended as the last band, pushing the true
        # (CSF-frequency) baseband to the second-to-last slot.
        baseband_idx = -2 if self.dc_term_enabled else -1
        if not self.baseband_freq_adapt:
            per_sband_w[:,:,0,baseband_idx] = self.baseband_weight[0:no_channels]
        if self.dc_term_enabled:
            per_sband_w[:,:,0,-1] = self.dc_weight[0:no_channels]

        #per_sband_w = torch.exp(interp1( self.quality_band_freq_log, self.quality_band_w_log, torch.log(torch.as_tensor(rho_band, device=self.device)) ))[:,None,None]

        Q_sc = self.lp_norm(Q_per_ch*per_ch_w*per_sband_w, self.beta_sch, dim=3, normalize=False)  # Sum across spatial bands

        is_image = (no_frames==1)
        t_int = self.image_int if is_image else 1.0 # Integration correction for images

        if not self.block_channels is None:
            Q_tc = self.lp_norm(Q_sc[self.block_channels[0:no_channels],...], self.beta_tch, dim=1, normalize=False)  # Sum across temporal and chromatic channels                
        else:
            Q_tc = self.lp_norm(Q_sc,     self.beta_tch, dim=1, normalize=False)  # Sum across temporal and chromatic channels

        if is_image:
            Q = Q_tc * t_int
        else:
            Q = self.lp_norm(Q_tc,     self.beta_t,   dim=2, normalize=True)   # Sum across frames

        Q = Q.squeeze()

        Q_JOD = self.met2jod(Q)            
        return Q_JOD

    # Convert contrast differences to JODs
    def met2jod(self, Q):

        # We could use 
        # Q_JOD = 10. - self.jod_a * Q**self.jod_exp
        # but it does not differentiate well near Q=0

        Q_t = 0.1
        jod_a_p = self.jod_a * (Q_t**(self.jod_exp-1.))

        Q_JOD = torch.empty_like(Q)
        Q_JOD[Q<=Q_t] = 10. - jod_a_p * Q[Q<=Q_t]
        Q_JOD[Q>Q_t] = 10. - self.jod_a * (Q[Q>Q_t]**self.jod_exp)
        return Q_JOD

    def compute_CSF( self, bb, logL_bkg, rho_band, batch_sz, all_ch, block_N_frames, rho_override=None ):
        rho = rho_band[bb] # Spatial frequency in cpd
        # rho_override: optional per-channel tensor (shape [all_ch]) of content-derived
        # spatial frequencies, used only for the baseband when baseband_freq_adapt=True.
        # When set, caching in csf.sensitivity is disabled (the value is content-dependent
        # and would never hit the cache anyway, and must keep its autograd graph intact).
        ch_height, ch_width = logL_bkg.shape[-2], logL_bkg.shape[-1]
        S_ref = torch.empty((batch_sz,all_ch,block_N_frames,ch_height,ch_width), device=self.device)
        if self.lum_adapt_reference: # For backward compatibility
            S_test = S_ref
        else:
            S_test = torch.empty((batch_sz,all_ch,block_N_frames,ch_height,ch_width), device=self.device)
        for cc in range(all_ch):
            tch = 0 if cc<3 else 1  # Sustained or transient
            cch = cc if cc<3 else 0 # Y, rg, yv
            this_rho = rho_override[cc] if rho_override is not None else rho
            use_cache = rho_override is None
            if self.lum_adapt_reference:
                # The sensitivity is computed assuming the adaptation to the luminance of the reference image only
                S_ref[:,cc:(cc+1),:,:,:] = self.csf.sensitivity(this_rho, self.omega[tch], logL_bkg[...,1:2,:,:,:], cch, self.csf_sigma, cache=use_cache) * 10.0**(self.sensitivity_correction/20.0)
            else:
                # The sensitivity is computed assuming the adaptation to the lumiance of the sustained channel of the test and reference images
                S_both = self.csf.sensitivity(this_rho, self.omega[tch], logL_bkg[...,0:2,:,:,:], cch, self.csf_sigma, cache=use_cache) * 10.0**(self.sensitivity_correction/20.0)
                S_test[:,cc:(cc+1),:,:,:] = S_both[:,0:1,:,:,:]
                S_ref[:,cc:(cc+1),:,:,:] = S_both[:,1:2,:,:,:]
        return (S_test, S_ref)

    def baseband_effective_rho(self, band_diff, ppd_band):
        # Differentiable content-adaptive spatial-frequency estimate for the baseband,
        # replacing the fixed 0.1 cpd anchor. Computed per channel as the power-spectrum-
        # weighted mean frequency (spectral centroid) of the test-reference difference in
        # that band. The DC bin is mapped to the CSF LUT's lowest tabulated frequency
        # (rather than excluded), so a pure luminance-pedestal difference (all energy at
        # DC) reproduces today's conservative behavior, and the estimate only shifts
        # upward when the baseband difference has genuine spatial structure.
        # band_diff: [batch, ch, frames, H, W]
        H, W = band_diff.shape[-2], band_diff.shape[-1]
        P = torch.fft.fft2(band_diff).abs()**2  # power spectrum, same shape as band_diff

        fy = (torch.fft.fftfreq(H, device=self.device) * ppd_band).abs()
        fx = (torch.fft.fftfreq(W, device=self.device) * ppd_band).abs()
        rho_grid = torch.sqrt(fy[:,None]**2 + fx[None,:]**2)  # [H,W], cyc/deg per FFT bin

        rho_min = 10.0**self.csf.log_rho[0]
        rho_max = 10.0**self.csf.log_rho[-1]
        rho_grid = rho_grid.clone()
        rho_grid[0,0] = rho_min

        P_sum = P.sum(dim=(0,2,3,4))  # aggregate over batch, frames, H, W - keep channel dim
        rho_num = (P * rho_grid).sum(dim=(0,2,3,4))
        rho_eff = rho_num / (P_sum + 1e-8)
        return rho_eff.clamp(min=rho_min, max=rho_max)  # [ch]

    # @torch.compile
    def process_block_of_frames(self, R, vid_sz, temp_ch, lpyr, is_image, capture_ref=None, ref_cache=None):
        # capture_ref: optional list - if given, appended per-band with a dict of
        #   {'S_ref': ...} (baseband) or {'S_ref': ..., 'R_p': ..., 'M': ...}
        #   (masking bands, filled in by apply_masking_model). Used to build a
        #   RefCache in precompute_reference(). Mutually exclusive with ref_cache.
        # ref_cache: optional RefCache - if given, S_ref/R_p/M are read from it
        #   instead of being recomputed from R's reference channels (compute_CSF is
        #   skipped entirely). Only valid for masking_model=="mult-ref" with a
        #   *_ref contrast mode, where S_test==S_ref and both are reference-only.
        #   Used by the *_with_cached_reference() fast path.
        # R[batch,channels,frames,width,height]
        # Channel order: test-sustained-Y, ref-sustained-Y, test-rg, ref-rg, test-yv, ref-yv, test-transient-Y, ref-transient-Y
        # Images do not have the two last channels
        #height, width, N_frames = vid_sz
        all_ch = 2+temp_ch
        batch_sz = R.shape[0]

        #torch.autograd.set_detect_anomaly(True)

        # if self.contrast=="log":
        #     R = lms2006_to_dkld65( torch.log10(R.clip(min=1e-5)) )

        # Perform Laplacian pyramid decomposition
        B_bands, L_bkg_pyr = lpyr.decompose(R)

        if self.debug: assert len(B_bands) == lpyr.get_band_count()

        if self.dump_channels:
            self.dump_channels.dump_lpyr(lpyr, B_bands)


        # if self.do_heatmap:
        #     Dmap_pyr_bands, Dmap_pyr_gbands = self.heatmap_pyr.decompose( torch.zeros([1,1,height,width], dtype=torch.float, device=self.device))

        # L_bkg_bb = [None for i in range(lpyr.get_band_count()-1)]

        rho_band = lpyr.get_freqs()
        rho_band[lpyr.get_band_count()-1] = 0.1 # Baseband

        Q_per_ch_block = None
        block_N_frames = R.shape[-3]
        n_bands_total = lpyr.get_band_count() + (1 if self.dc_term_enabled else 0)

        for bb in range(lpyr.get_band_count()):  # For each spatial frequency band

            is_baseband = (bb==(lpyr.get_band_count()-1))

            B_bb = lpyr.get_band(B_bands, bb) 
            T_f = B_bb[:,0::2,...] # Test
            R_f = B_bb[:,1::2,...] # Reference

            logL_bkg = lpyr.get_gband(L_bkg_pyr, bb)

            rho_override = None
            if is_baseband and self.baseband_freq_adapt:
                ppd_band = lpyr.ppd * (T_f.shape[-2] / lpyr.H)
                rho_override = self.baseband_effective_rho(T_f - R_f, ppd_band)

            band_capture = None if capture_ref is None else {}

            if ref_cache is not None:
                # S_test == S_ref under a *_ref contrast mode (both derived only from
                # the reference's L_bkg - see weber_g1_ref/weber_g0_ref in lpyr_dec.py),
                # so both are just the cached reference-only sensitivity - no LUT lookup.
                S_ref = ref_cache.bands[bb]['S_ref']
                S_test = S_ref
            else:
                # Compute CSF
                (S_test, S_ref) = self.compute_CSF( bb, logL_bkg, rho_band, batch_sz, all_ch, block_N_frames, rho_override=rho_override )
                if band_capture is not None:
                    band_capture['S_ref'] = S_ref.detach()

            if is_baseband:
                D = torch.abs(T_f*S_test-R_f*S_ref)
            else:
                # dimensions: [channel,frame,height,width]
                ref_cache_band = None if ref_cache is None else ref_cache.bands[bb]
                D = self.apply_masking_model(T_f, R_f, S_test, S_ref, capture_out=band_capture, ref_cache_band=ref_cache_band)

            if capture_ref is not None:
                capture_ref.append(band_capture)

            if Q_per_ch_block is None:
                Q_per_ch_block = torch.empty((batch_sz,all_ch, block_N_frames, n_bands_total), device=self.device)

            #assert (not D.isnan().any()) and (not D.isinf().any()) and (D>=0).all(), "Must not be nan and must be positive"

            Q_per_ch_block[:,:,:,bb] = self.lp_norm(D, self.beta, dim=(-2,-1), normalize=True, keepdim=False) # Pool across all pixels (spatial pooling)

            if self.do_heatmap:

                # We need to reduce the differences across the channels using the right weights
                # Weights for the channels: sustained, RG, YV, [transient]
                t_int = self.image_int if is_image else 1.0
                per_ch_w = self.get_ch_weights( all_ch ).view(-1,1,1,1) * t_int
                if is_baseband and not self.baseband_freq_adapt:
                    per_ch_w *= self.baseband_weight[0:all_ch].view(-1,1,1,1)

                D_chr = self.lp_norm(D*per_ch_w, self.beta_tch, dim=-4, normalize=False)  # Sum across temporal and chromatic channels
                self.heatmap_pyr.set_lband(bb, D_chr)

            if self.dump_channels:
                width = R.shape[-1]
                height = R.shape[-2]
                t_int = self.image_int if is_image else 1.0
                per_ch_w = self.get_ch_weights( all_ch ).view(-1,1,1,1) * t_int
                self.dump_channels.set_diff_band(width, height, lpyr.ppd, bb, D*per_ch_w, padding_type=self.spatial_padding)

        if self.dc_term_enabled:
            # Whole-frame, Weber-normalized opponent-channel difference, computed directly
            # on the full-resolution input (R), independent of the pyramid. Catches
            # large-area/near-uniform casts the CSF-frequency baseband term structurally
            # can't: their difference has ~no spatial structure (spectral centroid sits at
            # the CSF floor either way), and the pyramid crushes the whole frame down to a
            # tiny baseband grid before any of that machinery runs. No spatial extent, so
            # it's appended after pooling rather than fed through the heatmap/dump_channels
            # per-band visualization, which expects a spatial map per band.
            T_full = R[:,0::2,...] # Test, all channels, full resolution
            R_full = R[:,1::2,...] # Reference, all channels, full resolution
            # Shared adaptation luminance: mean of test+ref sustained-Y, same simplification
            # already used for the baseband's own Weber contrast (weber_g1 mode).
            L_bkg_full = torch.clamp(R[:,0:2,...].mean(dim=1, keepdim=True), min=0.01)
            D_dc = (torch.abs(T_full - R_full) / L_bkg_full).mean(dim=(-2,-1)) # [batch,ch,frames]
            Q_per_ch_block[:,:,:,-1] = D_dc

        if self.do_heatmap:
            heatmap_block = 1.-(self.met2jod( self.heatmap_pyr.reconstruct() )/10.)
        else:
            heatmap_block = None

        if self.dump_channels:
            self.dump_channels.dump_diff()

        return Q_per_ch_block, heatmap_block

    def mask_pool(self, C):
        # Cross-channel masking
        num_ch = C.shape[-4]
        if self.do_xchannel_masking:
            W = torch.reshape(self.xcm_pow, (4, 4))[:num_ch, :num_ch]
            M = torch.einsum('bkfhw,kc->bcfhw', C, W)
        else:
            cm_weights = torch.reshape( (2**self.xcm_weights), (1,4,1,1,1) )[:,:num_ch,...]
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

        D_max = self.d_max_pow

        return D_max * pow_neg( C_p, p ) / (0.2 + M)

    # a differentiable sign function
    def diff_sign(self, x):
        if x.requires_grad:
            return torch.tanh(10000.0 * x)
        else:
            return torch.sign(x)

    def apply_masking_model(self, T, R, S_test, S_ref, capture_out=None, ref_cache_band=None):
        # T - test contrast tensor T[batch,channel,frame,width,height]
        # R - reference contrast tensor
        # S - sensitivity
        # capture_out: optional dict - if given, the "ref" branch stores its
        #   reference-only intermediates (R_p, M) into it, detached. Used by
        #   precompute_reference() to build a RefCache.
        # ref_cache_band: optional dict from a previously captured RefCache band -
        #   if given, the "ref" branch reuses its R_p/M instead of recomputing them
        #   from R/S_ref. Used by the *_with_cached_reference() fast path. Only
        #   supported for masking_model=="mult-ref"; ignored otherwise.

        if self.masking_model in [ "mult-none", "add-transducer", "mult-transducer", "add-mutual", "mult-mutual", "mult-ref", "mult-mutual-old", "add-similarity", "mult-similarity", "mult-transducer-texture", "add-transducer-texture" ]:
            num_ch = T.shape[-4]
            if self.masking_model.startswith( "add" ):
                zero_tens = torch.as_tensor(0., device=T.device)
                ch_gain = self.ce_g * torch.reshape( torch.as_tensor( [1, 1.7, 0.237, 1.], device=T.device), (1, 4, 1, 1, 1) )[:,:num_ch,...]
                C_t = 1/S_test
                C_r = 1/S_ref
                T_p = self.diff_sign(T) * torch.maximum( (torch.abs(T)-C_t)*ch_gain + 1, zero_tens )
                R_p = self.diff_sign(R) * torch.maximum( (torch.abs(R)-C_r)*ch_gain + 1, zero_tens )
            else:
                if self.masking_model.endswith( "mutual-old" ):
                    T_p = T * S_test
                    R_p = R * S_ref
                else:
                    ch_gain = torch.reshape( torch.as_tensor( [1, 1.45, 1, 1.], device=T.device), (1, 4, 1, 1, 1) )[:,:num_ch,...]
                    T_p = T * S_test * ch_gain
                    if ref_cache_band is None:
                        R_p = R * S_ref * ch_gain
                    else:
                        R_p = ref_cache_band['R_p']

            if self.masking_model.endswith( "none" ):
                D = self.clamp_diffs(torch.abs(T_p-R_p))
            elif self.masking_model.endswith( "transducer" ):
                D = torch.abs(self.cm_transd(T_p)-self.cm_transd(R_p))                
            elif self.masking_model.endswith( "mutual" ):

                M_mm = self.phase_uncertainty(torch.min( torch.abs(T_p), torch.abs(R_p) ))
                p = self.mask_p
                q = self.mask_q[0:num_ch].view(num_ch,1,1,1)

                M = self.mask_pool(safe_pow(M_mm,q))

                #D_band = safe_pow(torch.abs(T_p - R_p),p)
                # k_c = self.k_c
                # D_clamped = k_c*D_band / (k_c + D_band)
                #D = D_clamped / (1 + M)
                D_u = safe_pow(torch.abs(T_p - R_p),p) / (1 + M)
                D = self.clamp_diffs( D_u )

            elif self.masking_model.endswith( "ref" ):

                p = self.mask_p
                if ref_cache_band is None:
                    M_mm = self.phase_uncertainty(torch.abs(R_p))
                    q = self.mask_q[0:num_ch].view(num_ch,1,1,1)
                    M = self.mask_pool(safe_pow(M_mm,q))
                else:
                    M = ref_cache_band['M']

                D_u = safe_pow(torch.abs(T_p - R_p),p) / (1 + M)
                D = self.clamp_diffs( D_u )

                if capture_out is not None:
                    capture_out['R_p'] = R_p.detach()
                    capture_out['M'] = M.detach()

            elif self.masking_model.endswith( "mutual-old" ):

                M_mm = self.phase_uncertainty(torch.min( torch.abs(T_p), torch.abs(R_p) ))
                p = self.mask_p
                q = self.mask_q[0:num_ch].view(1,num_ch,1,1,1)

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
    
                D_max = self.d_max_pow
                epsilon = D_max-1

                D = D_max - D_max*(2*torch.abs(T_p)*torch.abs(R_p)+epsilon)/(T_p_m*T_p_m + R_p_m*R_p_m + epsilon)

            # if not str(self.device) == "mps": # The reduction below does not work on MPS
            assert not (D.view(-1).isnan().any() or D.view(-1).isinf().any()), "Must not be NaN nor Inf"
        else:
            raise RuntimeError( f"Unknown masking model {self.masking_model}" )

        return D

    def clamp_diffs(self,D):
        if self.dclamp_type == "hard":
            Dc = torch.clamp(D, max=self.d_max_pow)
        elif self.dclamp_type == "soft":
            max_v = self.d_max_pow
            Dc = max_v * D / (max_v + D)
        elif self.dclamp_type == "none":
            Dc = D
        elif self.dclamp_type == "per_channel":
            num_ch = D.shape[0]
            max_v = self.d_max_pow[:num_ch,...].view(-1,1,1,1)
            Dc = max_v * D / (max_v + D)
        else:
            raise RuntimeError( f"Unknown difference clamping type {self.dclamp_type}" )

        return Dc


    def phase_uncertainty(self, M):
        # Blur only when the image is larger then the required pad size
        if self.pu_dilate != 0 and M.shape[-2]>self.pu_padsize and M.shape[-1]>self.pu_padsize:
            #M_pu = utils.imgaussfilt( M, self.pu_dilate ) * torch.pow(10.0, self.mask_c)
            H, W = M.shape[-2], M.shape[-1] # We need to reshape because the Gaussian does not work with 5D tensors
            M_pu = self.pu_blur.forward(M.view(-1,1,H,W)).view( M.shape[0:-2] + (H,W) ) * self.mask_c_pow
        else:
            M_pu = M * self.mask_c_pow
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
        max_v = self.d_max_pow
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

    def full_name(self):
        return "ColorVideoVDP"

    def quality_unit(self):
        return "JOD"

    def get_info_string(self):
        if self.display_name.startswith('standard_'):
            #append this if are using one of the standard displays
            standard_str = self.display_name
        else:
            standard_str = f'custom-display: {self.display_name}'

        if isinstance(self.display_photometry, tuple): # separate photometry for the test and reference
            L_black_test, L_refl_test = self.display_photometry[0].get_black_level()
            L_black_ref, L_refl_ref = self.display_photometry[1].get_black_level()
            return f'"{self.full_name()} v{self.version}, {self.pix_per_deg:.4g} [pix/deg], ' \
                f'Test: Lpeak={self.display_photometry[0].get_peak_luminance():.5g}, ' \
                f'Lblack={L_black_test:.4g}, Lrefl={L_refl_test:.4g} [cd/m^2]; '  \
                f'Reference: Lpeak={self.display_photometry[1].get_peak_luminance():.5g}, ' \
                f'Lblack={L_black_ref:.4g}, Lrefl={L_refl_ref:.4g} [cd/m^2]"' 
                
        else:
            L_black, L_refl = self.display_photometry.get_black_level()
            return f'"{self.full_name()} v{self.version}, {self.pix_per_deg:.4g} [pix/deg], ' \
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

        for cc in range(Q_per_ch.shape[1]): # for each temporal/chromatic channel
            for bb in range(Q_per_ch.shape[3]): # for each band
                fmap[f"t{cc}_b{bb}"] = Q_per_ch[:, cc,:,bb].tolist()

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
                # integers are never trained
                #parameters[key] = getattr(self, key).item()                
                continue
            elif isinstance(parameters[key], float):
                if torch.is_tensor(getattr(self, key)):
                    # np.float32 is not serializable
                    parameters[key] = np.float64(getattr(self, key).item())
                else:
                    parameters[key] = np.float64(getattr(self, key))
            elif isinstance(parameters[key], list):
                parameters[key] = list(getattr(self, key).detach().cpu().numpy().astype(np.float64))

        parameters['__comment'] = comment
        parameters['calibration_date'] = date.today().strftime('%d/%m/%Y')

        with open(fname, 'w') as f:
            json.dump(parameters, f, indent=4)


    # Export the visualization of distortions over time
    def export_distogram(self, stats, fname, jod_max=None, base_size=6):
        # Q_per_ch[batch,channel,frame,sp_band]
        Q_per_ch = torch.as_tensor( stats['Q_per_ch'], device=self.device )
        batch_no = Q_per_ch.shape[0]
        if batch_no != 1:
            raise cvvdp_exception( 'Exporting distograms in batch mode is not supported' )
        ch_no = Q_per_ch.shape[1]

        is_image = (Q_per_ch.shape[2]==1)

        # Note: with dc_term_enabled, Q_per_ch has one extra trailing band (the DC term)
        # not represented in stats['rho_band'] used below for axis labels; this
        # visualization is not band-axis-accurate in that mode.
        baseband_idx = -2 if self.dc_term_enabled else -1
        if not self.baseband_freq_adapt:
            Q_per_ch[:,:,:,baseband_idx] *= self.baseband_weight[0:ch_no].view(-1,1)
        if self.dc_term_enabled:
            Q_per_ch[:,:,:,-1] *= self.dc_weight[0:ch_no].view(-1,1)
        Q_per_ch *= self.get_ch_weights(ch_no)*ch_no
        dmap = (10. - self.met2jod(Q_per_ch)).cpu().numpy()

        if jod_max is None:
            jod_max = math.ceil(dmap.max())
        
        dmap /= jod_max

        fps = stats['frames_per_second']
        band_no = Q_per_ch.shape[3]
        frame_no = Q_per_ch.shape[2]
        rho_band = stats['rho_band']
        band_labels = [f"{val:.2f}" for val in np.flip(rho_band)[::2]]
        band_labels[0] = "BB"

        if not has_matplotlib:
            raise RuntimeError( 'matplotlib is missing. Please install it before exporting distograms.')
            
        fig, axs = plt.subplots(nrows=ch_no, figsize=(base_size*frame_no/60+1, base_size))

        ch_labels = ["A-sust", "RG", "YV", "A-trans"]
        cmap = plt.colormaps["plasma"]

        for kk in range(ch_no):
            dmap_ch = np.flip(np.transpose(dmap[0,kk,:,:].clip(0.,1.)),axis=0)
            axs[kk].imshow(dmap_ch, cmap=cmap, aspect="auto" )
            axs[kk].set_ylabel( ch_labels[kk] )
            axs[kk].yaxis.set_major_locator(ticker.FixedLocator(range(0,len(band_labels)*2,2)))
            axs[kk].yaxis.set_minor_locator(ticker.MultipleLocator(1.0))
            axs[kk].set_yticklabels(band_labels)
            if kk==(ch_no-1) and not is_image:
                axs[kk].xaxis.set_major_formatter(lambda x, pos: str(int(x/fps*1000)))
                axs[kk].set_xlabel( 'Time [ms]')
                axs[kk].xaxis.set_minor_locator(ticker.MultipleLocator(1.0))
            else:
                axs[kk].set_xticks([])

        if is_image:
            plt.subplots_adjust(bottom=0.1, right=0.5, top=0.9)
            cax = plt.axes([0.725, 0.1, 0.125, 0.8])
        else:
            plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
            cax = plt.axes([0.925, 0.1, 0.025, 0.8])
        
        plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, jod_max), cmap=cmap), cax=cax, cmap=cmap)

        # fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
        #             ax=axs[0], label="JODs")

        plt.savefig( fname, bbox_inches='tight' )  

        # fig.show()
        # plt.waitforbuttonpress()        
        
    # # Visualize the local contrast pyramid
    # def visualize_lpyr(self, test_cont, reference_cont, dim_order="BCFHW", frames_per_second=0, met_colorspace='DKLd65'):
    #     vid_source = video_source_array( test_cont, reference_cont, frames_per_second, dim_order=dim_order, display_photometry=self.display_photometry )
    #     vid_sz = vid_source.get_video_size() # H, W, F
    #     height, width, N_frames = vid_sz
    #     if self.lpyr is None or self.lpyr.W!=width or self.lpyr.H!=height:
    #         if self.contrast.startswith("weber"):
    #             self.lpyr = weber_contrast_pyr(width, height, self.pix_per_deg, self.device, contrast=self.contrast)
    #         elif self.contrast.startswith("log"):
    #             self.lpyr = log_contrast_pyr(width, height, self.pix_per_deg, self.device, contrast=self.contrast)
    #         else:
    #             raise RuntimeError( f"Unknown contrast {self.contrast}" )    
    #     R = torch.empty((1, 6, 1, height, width), device=self.device)
    #     if self.contrast=="log":
    #         met_colorspace='logLMS_DKLd65'
    #     else:
    #         met_colorspace='DKLd65' # This metric uses DKL colourspaxce with d65 whitepoint
    #     R[:,0::2, :, :, :] = vid_source.get_test_frame(0, device=self.device, colorspace=met_colorspace)
    #     R[:,1::2, :, :, :] = vid_source.get_reference_frame(0, device=self.device, colorspace=met_colorspace)
    #     B_bands, L_bkg_pyr = self.lpyr.decompose(R[0,...])

    #     test_pyr = []
    #     ref_pyr = []
    #     for bb in range(self.lpyr.get_band_count()):  # For each spatial frequency band
    #         B_bb = self.lpyr.get_band(B_bands, bb) 
    #         test_pyr.append(B_bb[0::2,...]) # Test
    #         ref_pyr.append(B_bb[1::2,...]) # Reference

    #     return test_pyr, ref_pyr
    
    # # Visualize the pyramids
    # def visualize_pyr(self, test_cont, reference_cont, dim_order="BCFHW", frames_per_second=0, keep_gaussian=False, met_colorspace='DKLd65'):
    #     vid_source = video_source_array( test_cont, reference_cont, frames_per_second, dim_order=dim_order, display_photometry=self.display_photometry )
    #     vid_sz = vid_source.get_video_size() # H, W, F
    #     height, width, N_frames = vid_sz
    #     if self.lpyr is None or self.lpyr.W!=width or self.lpyr.H!=height:
    #         self.lpyr = lpyr_dec_2(width, height, self.pix_per_deg, self.device, keep_gaussian=keep_gaussian)
    #     R = torch.empty((1, 6, 1, height, width), device=self.device)
    #     if self.contrast=="log":
    #         met_colorspace='logLMS_DKLd65'
    #     else:
    #         met_colorspace='DKLd65' # This metric uses DKL colourspaxce with d65 whitepoint
    #     R[:,0::2, :, :, :] = vid_source.get_test_frame(0, device=self.device, colorspace=met_colorspace)
    #     R[:,1::2, :, :, :] = vid_source.get_reference_frame(0, device=self.device, colorspace=met_colorspace)
        
    #     _, _ = self.lpyr.decompose(R[0,...])

    #     return self.lpyr

    # # Visualize the color-encoded frame
    # def visualize_encoded_frame(self, test_cont, reference_cont, dim_order="BCFHW", frames_per_second=0, keep_gaussian=False, met_colorspace='DKLd65'):
    #     vid_source = video_source_array( test_cont, reference_cont, frames_per_second, dim_order=dim_order, display_photometry=self.display_photometry )
    #     vid_sz = vid_source.get_video_size() # H, W, F
    #     height, width, N_frames = vid_sz
    #     if self.lpyr is None or self.lpyr.W!=width or self.lpyr.H!=height:
    #         self.lpyr = lpyr_dec_2(width, height, self.pix_per_deg, self.device, keep_gaussian=keep_gaussian)
    #     R = torch.empty((1, 6, 1, height, width), device=self.device)
    #     if self.contrast=="log":
    #         met_colorspace='logLMS_DKLd65'
    #     else:
    #         met_colorspace='DKLd65' # This metric uses DKL colourspaxce with d65 whitepoint
    #     R[:,0::2, :, :, :] = vid_source.get_test_frame(0, device=self.device, colorspace=met_colorspace)
    #     R[:,1::2, :, :, :] = vid_source.get_reference_frame(0, device=self.device, colorspace=met_colorspace)
        
    #     return R[0,...]


register_metric( cvvdp )

class cvvdp_0_5_6(cvvdp):
    def __init__(self, config_paths=[], **kwargs):
        path = os.path.join(os.path.dirname(__file__), "vvdp_data", 'cvvdp_parameters-0_5_6.json' )
        config_paths.insert(0,path)
        super().__init__(**kwargs, config_paths=config_paths)

register_metric( cvvdp_0_5_6 )
