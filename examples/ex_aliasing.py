# This example shows how to use python interface to run ColorVideoVDP directly on video files

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import glob
import time

import pycvvdp

def aliasing_example( metric_class = pycvvdp.cvvdp, device=None ):
    display_name = 'sdr_fhd_24'
    media_folder = os.path.join(os.path.dirname(__file__), '..',
                                'example_media', 'aliasing')

    ref_file = os.path.join(media_folder, 'ferris-ref.mp4')
    TST_FILEs = glob.glob(os.path.join(media_folder, 'ferris-*-*.mp4'))

    metric = metric_class(display_name=display_name, device=device, heatmap=None)

    res = []
    for tst_fname in TST_FILEs:

        vs = pycvvdp.video_source_file( tst_fname, ref_file, display_photometry=display_name )

        start = time.time()
        Q_JOD, stats_static = metric.predict_video_source( vs )
        end = time.time()

        tst_label = os.path.basename(tst_fname)
        tst_time = end-start
        print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_label, Q_JOD, tst_time) )
        res.append( (tst_label, Q_JOD, tst_time) )

    return res

if __name__ == '__main__':
    aliasing_example( )