# This example shows how ColorVideoVDP varies the visibility of blue noise to account for luminance masking

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import ex_utils as utils
import argparse

from torchvision.transforms import GaussianBlur

from examples.ex_utils import lin2srgb

import pycvvdp

from pycvvdp.video_writer import VideoWriter


def tensor_to_numpy_image(T):
    if T.ndim==2: # Grayscale
        (h, w) = T.shape
        return torch.squeeze(T).view(h,w,1).tile((1,1,3)).cpu().numpy()
    else:
        return torch.squeeze(T).cpu().numpy()


def generate_video( test_vid_fname, ref_vid_fname ):
    tile_sz = 64
    L_peak = 100
    L_mean = 30

    tiles = 8
    C_mask = torch.logspace( math.log10(0.05), math.log10(0.8), tiles )

    w, h = tiles*tile_sz, tile_sz # Image width and height

    T_ref = torch.ones( (h,w) )

    # Band-pass noise
    sigma1 = 2
    kernel_size = 2 * int(4 * sigma1 + 0.5) + 1
    blur1 = GaussianBlur(kernel_size=kernel_size, sigma=sigma1)
    sigma2 = 6
    kernel_size = 2 * int(4 * sigma2 + 0.5) + 1
    blur2 = GaussianBlur(kernel_size=kernel_size, sigma=sigma2)
    noise_white = torch.rand((tile_sz, tile_sz))
    noise_bl = (blur1(noise_white.view(1,tile_sz,tile_sz)) - blur2(noise_white.view(1,tile_sz,tile_sz))).view(tile_sz,tile_sz)
    noise_bl /= noise_bl.abs().max()

    for kk in range(C_mask.numel()):
        pos = kk*tile_sz
        T_ref[:,pos:(pos+tile_sz)] = L_mean + L_mean*noise_bl*C_mask[kk]

    noise_white = torch.rand((tile_sz, tile_sz))
    noise_bl2 = (blur1(noise_white.view(1,tile_sz,tile_sz)) - blur2(noise_white.view(1,tile_sz,tile_sz))).view(tile_sz,tile_sz)
    noise_bl2 /= noise_bl2.abs().max()

    [xx, yy] = torch.meshgrid( torch.linspace( -1, 1, tile_sz), torch.linspace( -1, 1, tile_sz) )
    sigma = 0.5
    gauss_env = torch.exp( -(xx**2+yy**2)/(2*sigma**2) )

    fps = 30
    time_on_tile = 0.8
    time_moving = 0.4
    time_total = time_on_tile*tiles + time_moving*(tiles-1)
    frames = int(time_total*fps)

    with (VideoWriter( test_vid_fname, hdr_mode=False, fps=fps, codec='h265', verbose=False) as vw_t,
        VideoWriter( ref_vid_fname, hdr_mode=False, fps=fps, codec='h265', verbose=False) as vw_r):

        for ff in range(frames):
            T_test = T_ref.clone()

            time_stamp = ff/fps
            tile_index = int(time_stamp / (time_on_tile + time_moving))
            ts_tile = time_stamp - tile_index*(time_on_tile + time_moving)
            move_r = max(ts_tile - time_on_tile, 0)/time_moving
            pos = int(tile_index*tile_sz + move_r*tile_sz)

            C_test = 0.2
            T_test[:,pos:(pos+tile_sz)] = (T_test[:,pos:(pos+tile_sz)] + L_mean*noise_bl2*gauss_env*C_test).clamp( 0.001, L_peak )

            vw_t.write_frame_rgb(tensor_to_numpy_image(lin2srgb(T_test/L_peak)))
            vw_r.write_frame_rgb(tensor_to_numpy_image(lin2srgb(T_ref/L_peak)))

def run_cvvdp_on_video( test_vid_fname, ref_vid_fname, heatmap_vid_fname ):
    display_name = 'standard_fhd'
    metric = pycvvdp.cvvdp(display_name=display_name, heatmap="supra-threshold", heatmap_file=heatmap_vid_fname)

    vs = pycvvdp.video_source_file( test_vid_fname, ref_vid_fname, display_photometry=display_name )

    Q_JOD, stats_static = metric.predict_video_source( vs )    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of using a custom video_source")
    parser.add_argument("--recompute", action='store_true', default=False, help="Regenerate the video files")
    args = parser.parse_args()

    test_vid_fname = 'contrast_masking_test.mp4'
    ref_vid_fname = 'contrast_masking_reference.mp4'
    heatmap_vid_fname = 'contrast_masking_heatmap.mp4'

    if not os.path.isfile( test_vid_fname ) or not os.path.isfile( ref_vid_fname ) or args.recompute:
        generate_video( test_vid_fname, ref_vid_fname )
    else:
        print( f"Using the precomputed videos from {test_vid_fname} and {ref_vid_fname}." )

    run_cvvdp_on_video( test_vid_fname, ref_vid_fname, heatmap_vid_fname )


