# This example shows how to call ColorVideoVDP on multiple images of the same size (a batch)
# Processing multiple images in a single batch can substantially speed up computation. 

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import ex_utils as utils
import pycvvdp

debug = False

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

I_test_both = np.concatenate( (I_test_noise[np.newaxis,...], I_test_blur[np.newaxis,...]), axis=0 )
I_ref_both = np.concatenate( (I_ref[np.newaxis,...], I_ref[np.newaxis,...]), axis=0 )

# metric = pycvvdp.cvvdp(display_name='standard_4k')
# metric = pycvvdp.cvvdp_ml_saliency(display_name='standard_4k')
metric = pycvvdp.cvvdp_ml_transformer(display_name='standard_4k')

# predict() method can handle numpy ndarrays or PyTorch tensors. The data
# type should be float32, int16 or uint8.
# Channels can be in any order, but the order must be specified as a dim_order parameter. 
# Here the dimensions are (Height,Width,Color)
Q_JOD, stats = metric.predict( I_test_both, I_ref_both, dim_order="BHWC" )

print( f'Noise - Quality: {Q_JOD[0]:.3f} JOD' )
print( f'Blur - Quality: {Q_JOD[1]:.3f} JOD' )
