# This example shows how to call ColorVideoVDP from python to predict quality for a pair of test and 
# reference images, stored as numpy arrays.

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import examples.ex_utils as utils
import pycvvdp
import time

'''
Results of current version (for reference):
Noise - Quality: 8.955 JOD
Blur - Quality: 8.514 JOD
'''

def simple_image_example( metric_class = pycvvdp.cvvdp, debug=False, device=None ):

    I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))

    noise_fname = os.path.join('example_media', 'wavy_facade_noise.png')
    if os.path.isfile(noise_fname) and debug:
        I_test_noise = pycvvdp.load_image_as_array( noise_fname )
    else:
        std = np.sqrt(0.003)
        I_test_noise = utils.imnoise(I_ref, std)

    blur_fname = os.path.join('example_media', 'wavy_facade_blur.png')
    if os.path.isfile(blur_fname) and debug:
        I_test_blur = pycvvdp.load_image_as_array( blur_fname )
    else:
        sigma = 2
        I_test_blur = utils.imgaussblur(I_ref, sigma)

    # You can initilize a metric of the specific class 
    # metric = pycvvdp.cvvdp(display_name='standard_4k', heatmap='threshold')
    # Here, we want to be able to change the metric class
    metric = metric_class(display_name='standard_4k', heatmap='none', device=device)

    # metric = pycvvdp.cvvdp_ml_saliency(display_name='standard_4k')
    # metric = pycvvdp.cvvdp_ml_transformer(display_name='standard_4k')

    res = []
    # predict() method can handle numpy ndarrays or PyTorch tensors. The data
    # type should be float32, int16 or uint8.
    # Channels can be in any order, but the order must be specified as a dim_order parameter. 
    # Here the dimensions are (Height,Width,Color)
    
    start = time.time()
    Q_JOD_noise, stats_noise = metric.predict( I_test_noise, I_ref, dim_order="HWC" )
    end = time.time()

    tst_time = end-start
    noise_str = f'Noise - Quality: {Q_JOD_noise:.3f} JOD ({tst_time:.2f} secs to complete)'
    print( noise_str )
    res.append( ('Noise', Q_JOD_noise, tst_time))

    start = time.time()
    Q_JOD_blur, stats_blur = metric.predict( I_test_blur, I_ref, dim_order="HWC" )
    end = time.time()

    tst_time = end-start
    blur_str = f'Blur - Quality: {Q_JOD_blur:.3f} JOD ({tst_time:.2f} secs to complete)'
    print( blur_str )
    res.append( ('Blur', Q_JOD_blur, tst_time))

    return res

if __name__ == '__main__':
    simple_image_example( )