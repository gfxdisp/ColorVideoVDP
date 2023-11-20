# This example shows how to use python interface to run FovVideoVDP directly on video files
import os
import glob
from tabnanny import verbose
import time

import pycvvdp
import logging

display_name = 'eizo_CG3146-XR-DAVID'

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

# ref_file = os.path.join(media_folder, 'Emojis_reference_Level001.mp4')
# TST_FILEs = glob.glob(os.path.join(media_folder, 'Emojis_DUC_Level003.mp4'))

# media_folder = 'S:\\Datasets\\LIVEHDR\\train'
# ref_file = os.path.join(media_folder, '4k_ref_CenterPanorama.mp4')
# TST_FILEs = glob.glob(os.path.join(media_folder, '4k_3M_CenterPanorama.mp4'))

# ref_file = os.path.join(media_folder, 'Phone_reference_Level001.mp4')
# TST_FILEs = glob.glob(os.path.join(media_folder, 'Phone_CSub_Level003.mp4'))

video="VR"
#distortion = "WGNU_Level003"
#distortion = "CSub_Level003"
distortion = "LSNU_Level003"

# ref_file = os.path.join(media_folder, video + '_reference_Level001.mp4')
# TST_FILEs = glob.glob(os.path.join(media_folder, video + '_' + distortion + '.mp4'))

ref_file = media_folder + "/" +  video + '_reference_Level001.mp4'
TST_FILEs = glob.glob(media_folder + "/" + video + '_' + distortion + '.mp4')

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)

config_paths = [media_folder, "../metric_configs/cvvdp_overconstancy/cvvdp_parameters.json"]

cvvdp = pycvvdp.cvvdp(display_name=display_name, heatmap="raw", config_paths=config_paths)
cvvdp.debug = True

for tst_fname in TST_FILEs:

    vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name, frames=60, verbose=False, config_paths=config_paths )

    start = time.time()
    Q_JOD_static, stats_static = cvvdp.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )

    cvvdp.export_distogram( stats_static, video + '_' + distortion + '_distogram.pdf', jod_max=10, base_size=3.5 )
