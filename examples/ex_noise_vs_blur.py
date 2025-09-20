# This example shows how ColorVideoVDP could be use for perceptual optimization. 
#
# The example simulates a camera that is affected by noise and motion blur (hand shake). 
# Longer exposure time can reduce amount of noise, but it will introduce blur due to hand motion. 
# The example shows how to find the exposure time that gives the best image quality.
#
# The predictions of ColorVideoVDP are compared to those of PSNR. ColorVideoVDP shows the highest predictions for the image that looks the best. This is not the case for PSNR.
#
# The example packs all the exposure times into a batch dimension - this allows for better parallelization and shorter exposure times, but requires more memory. 


# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import ex_utils as utils
import pycvvdp
import math

#from gfxdisp.pfs.pfs_torch import pfs_torch

def camera_model( I, t, g=1 ):

    # Simulate hand-shake blur
    vel = 6 # camera velocity in pixels per second
    k_sz = int(math.ceil(t*vel))
    kernel = torch.ones((1,1,1,k_sz), device=I.device)
    if k_sz>1:
        kernel[...,-1] = torch.frac(t*vel) # the last element is a fractional part 
    kernel = kernel/kernel.sum()
    I_chw = I.permute( (2,0,1) ).view( (I.shape[2], 1, I.shape[0], I.shape[1]) )
    I_blur = torch.nn.functional.conv2d( I_chw, kernel, padding='same' )
    I_blur = I_blur.view((I.shape[2], I.shape[0], I.shape[1])).permute( (1,2,0) )

    # Simulate camera noise
    a = 0.01
    b = 0.001
    n_std = torch.sqrt( I_blur*t*a + b )    
    I_n = (I_blur*t + torch.randn_like(I_blur)*n_std)/t

    return I_n


I_de = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))


# We will input linear colorspace images (hence EOTF='linear') and assume an SDR display
Y_disp_peak = 200
disp_photo = pycvvdp.vvdp_display_photo_eotf(Y_peak=Y_disp_peak, contrast=1000, EOTF='linear', E_ambient=10)

metric = [None, None]
#metric[0] = pycvvdp.cvvdp_ml_transformer(display_name='standard_4k', display_photometry=disp_photo)
metric[0] = pycvvdp.cvvdp(display_name='standard_4k', display_photometry=disp_photo)
metric[1] = pycvvdp.psnr_rgb(display_name='standard_4k', display_photometry=disp_photo)
device = metric[0].device

# We need the image to be in a linear space. Note that this is a 16-bit image
gamma = 2.2
I_lin = (torch.from_numpy(I_de.astype(np.float32)).to(device=device)/(2**16-1))**gamma

t = torch.logspace( -1, 1, 8 )

I_test = torch.empty( (t.numel(),) + I_lin.shape, device=device)
I_ref = torch.tile( I_lin, (t.numel(),1,1,1) )

for kk in range(t.numel()):
    I_test[kk,:,:,:] = camera_model( I_lin, t[kk] )

Q_JOD = [None,] * len(metric)
for kk in range(len(metric)):
    # Note that we multiply by Y_disp_peak, as with linear EOTF we must provide absolute values
    #with torch.no_grad():
    Q_JOD[kk], stats = metric[kk].predict( I_test*Y_disp_peak, I_ref*Y_disp_peak, dim_order="BHWC" )

for kk in range(t.numel()):
    print( f't={t[kk]:.4f}, quality= {Q_JOD[0][kk]:.4f} JOD' )

# plt.plot(t.numpy(), Q_JOD.cpu().numpy(), '-o')

# # plt.savefig('results.png')
# plt.show()

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(4, 4)

for i in range(8):
    ax = fig.add_subplot(gs[int(i/4), i%4])    
    img = (I_test[i,...].clamp(0,1)**(1/gamma)).cpu().numpy()
    ax.imshow( img, vmin=0, vmax=1 )
    ax.set_title(f't={t[i]:.4f}')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

# Bottom two rows: one subplot spans all columns
for kk in range(len(metric)):
    ax_bottom = fig.add_subplot(gs[2+kk, :])
    ax_bottom.plot(t.numpy(), Q_JOD[kk].cpu().numpy(), '-o')
    ax_bottom.grid(which='major', linestyle='-')
    ax_bottom.grid(which='minor', linestyle='--')
    ax_bottom.set_xscale('log')
    ax_bottom.set_xlabel('Exposure time')
    ax_bottom.set_ylabel(f'{metric[kk].short_name()} [{metric[kk].quality_unit()}]')

plt.tight_layout()
plt.show()
