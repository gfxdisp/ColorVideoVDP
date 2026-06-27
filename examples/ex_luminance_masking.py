# This example shows how ColorVideoVDP varies the visibility of blue noise to account for luminance masking

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import ex_utils as utils

from torchvision.transforms import GaussianBlur

from examples.ex_utils import lin2srgb

import pycvvdp

L_peak = 100
w, h = 1024, 256 # Image width and height

T_ref = torch.ones( (h,1) ) * torch.logspace( -1, math.log10(L_peak), w )

# High-pass (blue) noise
sigma = 3
kernel_size = 2 * int(4 * sigma + 0.5) + 1
blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
noise = (torch.rand_like(T_ref)*0.8).clamp(0,1)
noise = noise - blur(noise.view(1,h,w)).view(h,w)

T_test = T_ref + T_ref * noise

# We use geometry of FHDR SDR 24" display, but ignore its photometric
# properties and instead use linear luminance EOTF. Linear EOTF
# will pass absolute values to the metric after clipping them to display limits
# (its peak and black level) and adding the screen reflections. 
disp_photo = pycvvdp.vvdp_display_photo_eotf(L_peak, contrast=1000000, source_colorspace='luminance', E_ambient=0)
metric = pycvvdp.cvvdp(display_name='standard_fhd', display_photometry=disp_photo, heatmap="supra-threshold")

Q_JOD_noise, stats_noise = metric.predict( T_test, T_ref, dim_order="HW" )
noise_str = f'Quality: {Q_JOD_noise:.3f} JOD'
print( noise_str )

# heatmap is BCFHW - select the first batch, frame and permute colour to the last dim
heatmap = (stats_noise["heatmap"][0,:,0,:,:].permute([1,2,0])*255).to(torch.uint8).cpu().numpy()

fig, axs = plt.subplots( 3, 1, figsize=(8,8) )

axs[0].imshow( (lin2srgb(T_ref/L_peak)).numpy(), cmap='gray', vmin=0, vmax=1 )
axs[0].set_title( 'Reference image' )
axs[0].axis('off')

axs[1].imshow( (lin2srgb(T_test/L_peak)).numpy(), cmap='gray', vmin=0, vmax=1 )
axs[1].set_title( 'Test image' )
axs[1].axis('off')

axs[2].imshow( heatmap, vmin=0, vmax=1 )
axs[2].set_title( "ColorVideoVDP heatmap" )
axs[2].axis('off')

plt.tight_layout()
plt.show()


