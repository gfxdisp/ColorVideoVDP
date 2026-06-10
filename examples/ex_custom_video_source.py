# This example shows how to create a custom video source filter that can introduce distortions on the fly. 
# The example computes a surface showing how the quality changes when the resolution and refresh rate of 
# the video is varied.

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import time

from itertools import product

import torch
import torch.nn.functional as F
from torch.functional import Tensor
import math
import numpy as np

from tqdm import tqdm
import pycvvdp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch

import argparse 

#from pycvvdp.video_source import video_source, video_source_filter

# Floor function that should be robust to the floating point precision issues
def safe_floor(x):
    x_f = math.floor(x)
    return x_f if (x-x_f)<(1-1e-6) else x_f+1


# A custom video source that generates distortions on the fly
class vs_spatiotemporal_dist(pycvvdp.video_source_filter):

    def __init__(self, vs: pycvvdp.video_source, pixels_per_degree: float, resolution_scale: float, frame_rate_scale: float, velocity_dps: float):
        super().__init__(vs)
        self.resolution_scale = resolution_scale
        self.frame_rate_scale = frame_rate_scale
        self.velocity_dps = velocity_dps
        self.ppd = pixels_per_degree

    def get_reference_frame( self, frame_idx, device, colorspace ) -> Tensor:
        frame = super().get_reference_frame( frame_idx, device, colorspace )

        velocity_pps = self.ppd * self.velocity_dps # Velocity in pixels per second
        velocity_ppf = velocity_pps / self.get_frames_per_second() # Velocity in pixels per frame
        motion_pos_x = int(round(velocity_ppf * float(frame_idx)))
            
        # Shift the content of the tensor by 'motion_pos_x' columns in the right direction. The content that is pushed to the right is wrapped around and appears on the left side of the tensor (roll).
        return torch.roll(frame, motion_pos_x, dims=-1)


    def get_test_frame( self, frame_idx, device, colorspace ) -> Tensor:
    
        # Frame index after temporal resampling
        resample_frame_idx = int(safe_floor(float(frame_idx)*float(self.frame_rate_scale)+0.5)/float(self.frame_rate_scale))
        resample_frame_idx = min( resample_frame_idx, self.get_frame_count()-1 )
        # print(f"frame: {frame_idx} resampled: {resample_frame_idx} fr_scale: {self.frame_rate_scale}")
        
        frame=super().get_test_frame( resample_frame_idx, device, colorspace )    
        
        velocity_pps = self.ppd * self.velocity_dps # Velocity in pixels per second
        velocity_ppf = velocity_pps / self.get_frames_per_second() # Velocity in pixels per frame
        motion_pos_x = int(round(velocity_ppf * float(resample_frame_idx)))
            
        # Shift the content of the tensor by 'motion_pos_x' columns in the right direction. The content that is pushed to the right is wrapped around and appears on the left side of the tensor (roll).
        frame = torch.roll(frame, motion_pos_x, dims=-1)

        # Simulate a lower resolution by downscaling and upscaling (with the nearest-neighbor)
        ds_size = (frame.shape[-3], round(frame.shape[-2]*self.resolution_scale), round(frame.shape[-1]*self.resolution_scale))
        downsampled_frame = F.interpolate(frame, size=ds_size, mode='nearest-exact')
        upsampled_frame = F.interpolate(downsampled_frame, size=(frame.shape[-3], frame.shape[-2], frame.shape[-1]), mode='nearest-exact')

        return upsampled_frame

from pycvvdp.dm_preview_metric import *


def compute_spatiotemp_quality( result_file, metric_class = pycvvdp.cvvdp, device=None ):

    display_name = 'standard_4k'

    # reference_file = os.path.join(os.path.dirname(__file__), '..',
    #                             'example_media', 'aliasing', 'ferris-ref.mp4')

    reference_file = os.path.join(os.path.dirname(__file__), '..',
                                'example_media', 'cyberpunk_crop.mp4')

    RESOLUTIONs = [1.0, 0.9, 0.75, 0.5, 0.25 ] #np.linspace( 0.25, 1, 11 ).tolist()
    FRAME_RATEs = [1.0, 60/120, 40/120, 30/120, 15/120] #np.linspace( 0.25, 1, 11 ).tolist()
    VELOCITYs = [0, 5] # Velocity in degrees per second    

    if False: # dm_preview_sbs will generate video with the test and reference videos - great way to debug a custom video_source
        metric = dm_preview_sbs(display_name=display_name, device=device)
        vs_source = pycvvdp.video_source_file( reference_file, reference_file, display_photometry=display_name, preload=True )
        dg = vvdp_display_geometry.load( display_name )
        vs = vs_spatiotemporal_dist( vs_source, dg.get_ppd(), 1.0, 30/120, 5 )
        metric.base_fname = 'spatiotemp_example'
        Q_JOD, stats_static = metric.predict_video_source( vs )        

    metric = metric_class(display_name=display_name, device=device, heatmap=None, quiet=True)

    # We want preload=True to load the video once and reuse across iterations and for random access
    vs_source = pycvvdp.video_source_file( reference_file, reference_file, display_photometry=display_name, preload=True )

    base_res = (vs_source.get_video_size()[1], vs_source.get_video_size()[0])
    res_label = [ f"{round(base_res[0]*rs)}x{round(base_res[1]*rs)}" for rs in RESOLUTIONs]
    print( f"Base resolution {base_res}; tested resolutions: {res_label}" )

    base_fps = vs_source.get_frames_per_second()
    fr_label = [ f"{round(fs*base_fps)}" for fs in FRAME_RATEs]
    print( f"Base frame rate {base_fps}; tested frame rates: {fr_label} fps" )

    with open(result_file, 'wt') as fh:
        fh.write( 'res_scale, res_label, fr_scale, fr_label, velocity, Q_JOD\n' )
        start = time.time()
        total_it = len(RESOLUTIONs)*len(FRAME_RATEs)*len(VELOCITYs)
        for res_scale, fr_scale, velocity_dps in tqdm(product( RESOLUTIONs, FRAME_RATEs, VELOCITYs ), unit="it", total=total_it):

            vs = vs_spatiotemporal_dist( vs_source, metric.display_geometry.get_ppd(), res_scale, fr_scale, velocity_dps )

            Q_JOD, stats_static = metric.predict_video_source( vs )        
            if metric.device.type == 'cuda':
                torch.cuda.empty_cache() # Clear cache to avoid memory problems

            res_idx = RESOLUTIONs.index(res_scale)
            fr_idx = FRAME_RATEs.index(fr_scale)
            # print( f'Quality for resolution {res_scale} and frame rate {fr_scale}: {Q_JOD:.3f} JOD' )
            fh.write( f'{res_scale}, {res_label[res_idx]}, {fr_scale}, {fr_label[fr_idx]}, {velocity_dps}, {Q_JOD}\n' )

        end = time.time()
        tst_time = end-start
        print( f"All took {tst_time:.4f} secs to compute." )

def plot_spatiotemp_quality( result_file ):
    df = pd.read_csv(result_file, sep=',', skipinitialspace=True)

    tick_df_fr = df[['fr_scale', 'fr_label']].drop_duplicates().sort_values('fr_scale')
    tick_df_res = df[['res_scale', 'res_label']].drop_duplicates().sort_values('res_scale')

    velocities = sorted(df['velocity'].unique())
    n = len(velocities)

    colors = [cm.Accent(i / max(n - 1, 1)) for i in range(n)]

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(121, projection='3d')

    for color, vel in zip(colors, velocities):
        subset = df[df['velocity'] == vel]
        pivot = subset.pivot_table(index='fr_scale', columns='res_scale', values='Q_JOD')

        X, Y = np.meshgrid(pivot.columns.values, pivot.index.values)
        Z = pivot.values

        ax.plot_surface(X, Y, Z, color=color, edgecolor='k', linewidth=0.3, alpha=0.6)

    ax.set_xlabel('resolution')
    ax.set_ylabel('frame rate')
    ax.set_zlabel('Q_JOD')
    ax.set_title('Spatio-temporal quality (Q_JOD)')
    ax.set_xticks(tick_df_res['res_scale'], labels=tick_df_res['res_label'])    
    ax.set_yticks(tick_df_fr['fr_scale'], labels=tick_df_fr['fr_label'])    

    legend_handles = [Patch(facecolor=color, label=f'{vel} dps') for color, vel in zip(colors, velocities)]
    ax.legend(handles=legend_handles, title='velocity')

    ax2 = fig.add_subplot(122)

    res_scales = sorted(df['res_scale'].unique())
    res_extremes = [res_scales[0], res_scales[-1]]
    linestyles = ['--', '-']  # dashed for lowest, solid for highest res_scale

    for color, vel in zip(colors, velocities):
        subset = df[df['velocity'] == vel]
        for ls, res in zip(linestyles, res_extremes):
            row = subset[subset['res_scale'].round(6) == round(res, 6)].sort_values('fr_scale')
            label = f'{vel} dps, res={res:.2f}'
            ax2.plot(row['fr_scale'], row['Q_JOD'], color=color, linestyle=ls, marker='o', label=label)

    ax2.set_xlabel('frame rate')
    ax2.set_ylabel('Q_JOD')
    ax2.set_title('Q_JOD vs frame rate')
    ax2.legend(title='velocity / res_scale', fontsize='small')
    ax2.set_xticks(tick_df_fr['fr_scale'], labels=tick_df_fr['fr_label'])    

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of using a custom video_source")
    parser.add_argument("--recompute", action='store_true', default=False, help="Recompute the data and overwrite the existing data file")
    args = parser.parse_args()

    result_file = 'spatiotemp_results.csv'

    if not os.path.isfile( result_file ) or args.recompute:
        compute_spatiotemp_quality( result_file )
    else:
        print( f"Using the precomputed results from {result_file}." )
    plot_spatiotemp_quality( result_file )
