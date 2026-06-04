# This example shows how to call ColorVideoVDP on multiple videos of the same size (a batch)
# Processing multiple videos in a single batch can substantially speed up computation, but it
# requires more memory.

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import examples.ex_utils as utils
import pycvvdp
import time

def batch_of_video_example( metric_class = pycvvdp.cvvdp, device=None, debug = False ):

    I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))

    #noise_fname = os.path.join('example_media', 'wavy_facade_noise.png')

    std = np.sqrt(0.003)
    I_test_noise = utils.imnoise(I_ref, std)

    blur_fname = os.path.join('example_media', 'wavy_facade_blur.png')
    sigma = 2
    I_test_blur = utils.imgaussblur(I_ref, sigma)

    I_ref = np.repeat( I_ref[np.newaxis,...], 5, axis=0 )
    I_test_blur = np.repeat( I_test_blur[np.newaxis,...], 5, axis=0 )
    I_test_noise = np.repeat( I_test_noise[np.newaxis,...], 5, axis=0 )

    I_test_both = np.concatenate( (I_test_noise[np.newaxis,...], I_test_blur[np.newaxis,...]), axis=0 )
    I_ref_both = np.concatenate( (I_ref[np.newaxis,...], I_ref[np.newaxis,...]), axis=0 )

    metric = metric_class(display_name='standard_4k', device=device)

    # predict() method can handle numpy ndarrays or PyTorch tensors. The data
    # type should be float32, int16 or uint8.
    # Channels can be in any order, but the order must be specified as a dim_order parameter. 
    # Here the dimensions are (Height,Width,Color)
    start = time.time()
    Q_JOD, stats = metric.predict( I_test_both, I_ref_both, dim_order="BFHWC", frames_per_second=30 )
    end = time.time()
    tst_time = end-start

    # try:
    #     metric.export_distogram( stats, 'test.png' )
    # except pycvvdp.vq_exception as ex:
    #     print( f"Exception: {str(ex)}")

    print( f'Noise - Quality: {Q_JOD[0]:.3f} JOD' )
    print( f'Blur - Quality: {Q_JOD[1]:.3f} JOD' )

    res = []
    res.append( ('Noise', Q_JOD[0], tst_time))
    res.append( ('Blur', Q_JOD[1], tst_time))
    return res

if __name__ == '__main__':
    batch_of_video_example( )