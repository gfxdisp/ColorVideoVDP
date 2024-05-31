# This example shows how to run ColorVideoVDP from python on HDR images.

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import numpy as np
import matplotlib.pyplot as plt
import ex_utils as utils

import pycvvdp

'''
Results of current version (for reference):
Noise - Quality: 9.450 JOD
Blur - Quality: 8.696 JOD
'''

I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'nancy_church.hdr'))

noise_fname = os.path.join('example_media', 'wavy_facade_noise.png')
L_peak = 4000   # Peak luminance of an HDR display

# HDR images are often given in relative photometric units. They MUST be
# mapped to absolute amount of light emitted from the display. For that, 
# we map the peak value in the image to the peak value of the display,
# then we increase the brightness by 2 stops (*4). This is an arbitrary 
# choise (of colour grading/tone mapping), and different mapping could be used. 
I_ref = I_ref/I_ref.max() * L_peak * 4

# Add Gaussian noise of 20% contrast
I_test_noise = (I_ref + I_ref*np.random.randn(*I_ref.shape)*0.3).astype(I_ref.dtype)

I_test_blur = utils.imgaussblur(I_ref, 2)

# We use geometry of SDR 4k 30" display, but ignore its photometric
# properties and instead use a display model with linear EOTF. Linear EOTF
# will pass absolute values to the metric after clipping them to display limits
# (its peak and black level) and adding the screen reflections. 
# Note that many HDR images are in BT.709 color space, so no need to
# specify BT.2020. 
disp_photo = pycvvdp.vvdp_display_photo_eotf(L_peak, contrast=1000000, source_colorspace='BT.709', EOTF="linear", E_ambient=100)
metric = pycvvdp.cvvdp(display_name='standard_hdr_linear', display_photometry=disp_photo, heatmap='threshold')

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

# f, axs = plt.subplots(1, 2)
# axs[0].imshow( stats_noise['heatmap'][0,:,0,:,:].permute([1,2,0]).cpu().numpy() )
# axs[0].set_xticks([])
# axs[0].set_yticks([])
# axs[0].set_title(noise_str)
# axs[1].imshow( stats_blur['heatmap'][0,:,0,:,:].permute([1,2,0]).cpu().numpy() )
# axs[1].set_xticks([])
# axs[1].set_yticks([])
# axs[1].set_title(blur_str)

# f.show()
# plt.waitforbuttonpress()
