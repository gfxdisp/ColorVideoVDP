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

media_folder = '../color_metric_videos/'
# ref_file = os.path.join(media_folder, 'Business_reference_Level001.mp4')
# TST_FILEs = glob.glob(os.path.join(media_folder, 'Business_WGNU_Level003.mp4'))

ref_file = os.path.join(media_folder, 'Bonfire_reference_Level001.mp4')
TST_FILEs = glob.glob(os.path.join(media_folder, 'Bonfire_Blur_Level003.mp4'))

# media_folder = 'S:\\Datasets\\LIVEHDR\\train'
# ref_file = os.path.join(media_folder, '4k_ref_CenterPanorama.mp4')
# TST_FILEs = glob.glob(os.path.join(media_folder, '4k_3M_CenterPanorama.mp4'))


logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)

"""

print('CVVDP test')
cvvdp = pycvvdp.cvvdp(display_name=display_name)
cvvdp.debug = False
for tst_fname in TST_FILEs:
    
    vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name, frames=10 )

    start = time.time()
    Q_JOD_static, stats_static = cvvdp.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )


print('PU PSNR test')
psnr = pycvvdp.pu_psnr_y()
psnr.debug = False
for tst_fname in TST_FILEs:

    vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name, frames=10 )

    start = time.time()
    Q_JOD_static, stats_static = psnr.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )


print('E_ITP test')
eitp = pycvvdp.e_itp()
eitp.debug = False
for tst_fname in TST_FILEs:

    vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name, frames=10 )

    start = time.time()
    Q_JOD_static, stats_static = eitp.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )


print('E_Spatial ITP test')
esitp = pycvvdp.e_sitp(display_name=display_name)
esitp.debug = False
for tst_fname in TST_FILEs:

    vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name, frames=10 )

    start = time.time()
    Q_JOD_static, stats_static = esitp.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )


print('DE2000 test')
de00 = pycvvdp.de2000(display_name=display_name)
de00.debug = False
for tst_fname in TST_FILEs:

    vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name, frames=10 )

    start = time.time()
    Q_JOD_static, stats_static = de00.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )
    
    """
    
print('Spatial DE2000 test')
s_de00 = pycvvdp.s_de2000(display_name=display_name)
s_de00.debug = False
for tst_fname in TST_FILEs:

    vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name, frames=10 )

    start = time.time()
    Q_JOD_static, stats_static = s_de00.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )
    