# This example shows how to call ColorVideoVDP from python to predict quality for a pair of test and 
# reference images, stored as numpy arrays.

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import ex_utils as utils
import pycvvdp

'''
Results of current version (for reference):
Noise - Quality: 9.051 JOD
Blur - Quality: 7.951 JOD
'''

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

metric = pycvvdp.cvvdp(display_name='standard_4k', heatmap='threshold')

# predict() method can handle numpy ndarrays or PyTorch tensors. The data
# type should be float32, int16 or uint8.
# Channels can be in any order, but the order must be specified as a dim_order parameter. 
# Here the dimensions are (Height,Width,Colour)
Q_JOD_noise, stats_noise = metric.predict( I_test_noise, I_ref, dim_order="HWC" )
noise_str = f'Noise - Quality: {Q_JOD_noise:.3f} JOD'
print( noise_str )

Q_JOD_blur, stats_blur = metric.predict( I_test_blur, I_ref, dim_order="HWC" )
blur_str = f'Blur - Quality: {Q_JOD_blur:.3f} JOD'
print( blur_str )

# f, axs = plt.subplots(2, 2)
# axs[0][0].imshow( I_test_noise/(2**16 - 1) )
# axs[0][0].set_title('Test image with noise')
# axs[0][0].set_xticks([])
# axs[0][0].set_yticks([])
# axs[0][1].imshow( I_test_blur/(2**16 - 1) )
# axs[0][1].set_title('Test image with blur')
# axs[0][1].set_xticks([])
# axs[0][1].set_yticks([])
# axs[1][0].imshow( stats_noise['heatmap'][0,:,0,:,:].permute([1,2,0]).to(torch.float32).numpy() )
# axs[1][0].set_xticks([])
# axs[1][0].set_yticks([])
# axs[1][0].set_title(noise_str)
# axs[1][1].imshow( stats_blur['heatmap'][0,:,0,:,:].permute([1,2,0]).to(torch.float32).numpy() )
# axs[1][1].set_xticks([])
# axs[1][1].set_yticks([])
# axs[1][1].set_title(blur_str)

# f.show()
# plt.waitforbuttonpress()
