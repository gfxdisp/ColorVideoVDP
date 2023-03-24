# This example shows how to use python interface to run FovVideoVDP directly on video files
import os
import glob
from tabnanny import verbose
import time

import pycvvdp
import logging

display_name = 'eizo_CG3146'

# media_folder = 'S:\\Datasets\\XR-DAVID\\cache'
# ref_file = os.path.join(media_folder, 'Bonfire_reference_1920x1080_10b_444_709_30fps.yuv')
# TST_FILEs = glob.glob(os.path.join(media_folder, 'Bonfire_Blur_*.yuv'))

media_folder = '../datasets/XR-DAVID'
# ref_file = os.path.join(media_folder, 'Business_reference_Level001.mp4')
# TST_FILEs = glob.glob(os.path.join(media_folder, 'Business_WGNU_Level003.mp4'))

# ref_file = os.path.join(media_folder, 'Business_reference_Level001.mp4')
# TST_FILEs = glob.glob(os.path.join(media_folder, 'Business_DUC_Level003.mp4'))

# ref_file = os.path.join(media_folder, 'Snow_reference_Level001.mp4')
# TST_FILEs = glob.glob(os.path.join(media_folder, 'Snow_Dither_Level003.mp4'))

ref_file = os.path.join(media_folder, 'Emojis_reference_Level001.mp4')
TST_FILEs = glob.glob(os.path.join(media_folder, 'Emojis_DUC_Level003.mp4'))

# media_folder = 'S:\\Datasets\\LIVEHDR\\train'
# ref_file = os.path.join(media_folder, '4k_ref_CenterPanorama.mp4')
# TST_FILEs = glob.glob(os.path.join(media_folder, '4k_3M_CenterPanorama.mp4'))


pycvvdp.utils.config_files.set_config_dir(media_folder)
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)

cvvdp = pycvvdp.cvvdp(display_name=display_name, heatmap="raw")
cvvdp.debug = True

for tst_fname in TST_FILEs:

    vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name, frames=120, verbose=True )

    start = time.time()
    Q_JOD_static, stats_static = cvvdp.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )
