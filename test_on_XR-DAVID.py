# This example shows how to use python interface to run FovVideoVDP directly on video files
import os
import glob
import time

import pycvvdp
import pycvvdp.video_source_yuv

display_name = 'standard_4k'
media_folder = 'S:\\Datasets\\XR-DAVID\\cache'

ref_file = os.path.join(media_folder, 'Bonfire_reference_1920x1080_10b_444_709_30fps.yuv')
TST_FILEs = glob.glob(os.path.join(media_folder, 'Bonfire_Blur_*.yuv'))

cvvdp = pycvvdp.cvvdp(display_name=display_name)

for tst_fname in TST_FILEs:

    vs = pycvvdp.video_source_yuv.fvvdp_video_source_yuv_file( tst_fname, ref_file, display_photometry=display_name )

    start = time.time()
    Q_JOD_static, stats_static = cvvdp.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )
