# This example shows how to use python interface to run FovVideoVDP directly on video files
import os
import glob
import time

import pycvvdp
import logging

display_name = 'standard_4k'

# media_folder = 'S:\\Datasets\\XR-DAVID\\cache'
# ref_file = os.path.join(media_folder, 'Bonfire_reference_1920x1080_10b_444_709_30fps.yuv')
# TST_FILEs = glob.glob(os.path.join(media_folder, 'Bonfire_Blur_*.yuv'))

media_folder = 'S:\\Datasets\\XR-DAVID'
ref_file = os.path.join(media_folder, 'Dance_reference_Level001.mp4')
TST_FILEs = glob.glob(os.path.join(media_folder, 'Dance_Contrast_Level001.mp4'))

# media_folder = 'S:\\Datasets\\LIVEHDR\\train'
# ref_file = os.path.join(media_folder, '4k_ref_CenterPanorama.mp4')
# TST_FILEs = glob.glob(os.path.join(media_folder, '4k_3M_CenterPanorama.mp4'))


logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)

cvvdp = pycvvdp.cvvdp(display_name=display_name)
cvvdp.debug = True

for tst_fname in TST_FILEs:

    vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name, frames=30 )    

    start = time.time()
    Q_JOD_static, stats_static = cvvdp.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )
