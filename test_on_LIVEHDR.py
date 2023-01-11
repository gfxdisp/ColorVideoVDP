# This example shows how to use python interface to run FovVideoVDP directly on video files
import os
import glob
import time

import pycvvdp
import logging

import pycvvdp.utils

display_name = 'samsung_q90t_qled_livehdr_dark'

media_folder = '../datasets/LIVEHDR'
# TST_FILEs = glob.glob(os.path.join(media_folder, 'train', '1080p_6M_football4.mp4'))
# ref_file = os.path.join(media_folder, 'train', '4k_ref_football4.mp4')

TST_FILEs = glob.glob(os.path.join(media_folder, 'train', '720p_2.6M_firework.mp4'))
ref_file = os.path.join(media_folder, 'train', '4k_ref_firework.mp4')

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)

pycvvdp.utils.config_files.set_config_dir(media_folder)
cvvdp = pycvvdp.cvvdp(display_name=display_name)
cvvdp.debug = True

for tst_fname in TST_FILEs:

    vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name, frames=60, full_screen_resize='bicubic', resize_resolution=(3840, 2160) )

    start = time.time()
    Q_JOD_static, stats_static = cvvdp.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )
