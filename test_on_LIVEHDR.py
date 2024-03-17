# This example shows how to use python interface to run ColorVideoVDP directly on video files
import os
import glob
import time

import pycvvdp
import logging

import pycvvdp.utils

display_name = 'samsung_q90t_qled_livehdr_bright'

media_folder = '../datasets/LIVEHDR'
# TST_FILEs = glob.glob(os.path.join(media_folder, 'train', '1080p_6M_football4.mp4'))
# ref_file = os.path.join(media_folder, 'train', '4k_ref_football4.mp4')

TST_FILEs = glob.glob(os.path.join(media_folder, 'train', '4k_15M_NighTraffic.mp4'))
ref_file = os.path.join(media_folder, 'train', '4k_ref_NighTraffic.mp4')

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)

config_paths = [media_folder, "../metric_configs/cvvdp_mult_transducer_texture/cvvdp_parameters.json"]

cvvdp = pycvvdp.cvvdp(display_name=display_name, config_paths=config_paths)
cvvdp.debug = True

for tst_fname in TST_FILEs:

    vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name, frames=30, full_screen_resize='bicubic', resize_resolution=(3840, 2160), config_paths=config_paths )

    start = time.time()
    Q_JOD_static, stats_static = cvvdp.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )
