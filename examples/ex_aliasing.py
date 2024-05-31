# This example shows how to use python interface to run FovVideoVDP directly on video files

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import glob
import time

import pycvvdp

'''
Results of current version (for reference):
Quality for example_media/aliasing/ferris-bicubic-bicubic.mp4: 7.237 JOD (took 2.2799 secs to compute)
Quality for example_media/aliasing/ferris-bicubic-nearest.mp4: 7.096 JOD (took 1.3296 secs to compute)
Quality for example_media/aliasing/ferris-nearest-bicubic.mp4: 7.144 JOD (took 1.3378 secs to compute)
Quality for example_media/aliasing/ferris-nearest-nearest.mp4: 7.082 JOD (took 1.3284 secs to compute)
'''

display_name = 'sdr_fhd_24'
media_folder = os.path.join(os.path.dirname(__file__), '..',
                            'example_media', 'aliasing')

ref_file = os.path.join(media_folder, 'ferris-ref.mp4')
TST_FILEs = glob.glob(os.path.join(media_folder, 'ferris-*-*.mp4'))

metric = pycvvdp.cvvdp(display_name=display_name, heatmap=None)

for tst_fname in TST_FILEs:

    vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name )

    start = time.time()
    Q_JOD_static, stats_static = metric.predict_video_source( vs )
    end = time.time()

    print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )
