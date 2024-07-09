# This example shows how to run ColorVideoVDP on a pair of images that contain only luminance

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import numpy as np
import matplotlib.pyplot as plt
import ex_utils as utils

import pycvvdp

L_peak = 1000
L = 100 # Background luminance
w, h = 1920, 1080 # Image width and height

I_reference = np.ones( (h,w) ) * L
I_test = I_reference + I_reference*np.random.randn(*I_reference.shape)*0.3

# We use geometry of FHDR SDR 24" display, but ignore its photometric
# properties and instead use linear luminance EOTF. Linear EOTF
# will pass absolute values to the metric after clipping them to display limits
# (its peak and black level) and adding the screen reflections. 
disp_photo = pycvvdp.vvdp_display_photo_eotf(L_peak, contrast=1000000, source_colorspace='luminance', E_ambient=0)
metric = pycvvdp.cvvdp(display_name='standard_fhd', display_photometry=disp_photo)

# predict() method can handle numpy ndarrays or PyTorch tensors. The data
# type should be float32, int16 or uint8.
# Channels can be in any order, but the order must be specified as a dim_order parameter. 
# Here the dimensions are (Height,Width,Colour)
Q_JOD_noise, stats_noise = metric.predict( I_test, I_reference, dim_order="HW" )
noise_str = f'Noise - Quality: {Q_JOD_noise:.3f} JOD'
print( noise_str )
