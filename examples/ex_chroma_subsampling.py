import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import ex_utils as utils
import pycvvdp
from pycvvdp.colorspace import ColorTransform
from pycvvdp.video_source import reshuffle_dims
import torchvision.transforms as tt
import math

import cv2

# For debugging only
# from gfxdisp.pfs.pfs import pfs

# This is a 16-bit image, convert to float
I_ref_np16 = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
I_ref = I_ref_np16.astype(np.float32) / np.iinfo(I_ref_np16.dtype).max

# I_ref = reshuffle_dims( torch.as_tensor(I_ref_np), "HWC", "CFHW" )
# ct = ColorTransform()
# I_DKL = ct.rgb2colourspace(I_ref, "DKLd65")

# sigma = 4
# GB = tt.GaussianBlur(math.ceil(sigma*3)*2+1, sigma)
# for cc in range(1, 3):
#     I_DKL[cc,:,:,:] = GB(I_DKL[cc,:,:,:])

epsilon = 1e-4 # to avoid div by 0
I_Yxy = utils.im_ctrans( utils.im_ctrans( I_ref, 'srgb', 'rgb709' ) + epsilon, 'rgb709', 'Yxy' )

cvvdp = pycvvdp.cvvdp(display_name='standard_4k')
psnr_met = pycvvdp.pu_psnr_rgb2020()
psnr_met.set_display_model(display_name='standard_4k')

ss_factors = [1.5, 2, 4, 8, 16]
Q_JOD = [None] * len(ss_factors)
Q_PSNR = [None] * len(ss_factors)
I_test = [None] * len(ss_factors)

for kk in range(len(ss_factors)):
    dim = (I_ref.shape[1], I_ref.shape[0])
    dim_ss = (int(I_ref.shape[1]/ss_factors[kk]), int(I_ref.shape[0]/ss_factors[kk]))
    chroma_ss = cv2.resize(I_Yxy[:,:,1:3], dsize=dim_ss, interpolation=cv2.INTER_CUBIC)
    chroma_rec = cv2.resize(chroma_ss, dsize=dim, interpolation=cv2.INTER_CUBIC)
    I_Yxy[:,:,1:3] = chroma_rec

    I_test[kk] = utils.im_ctrans( I_Yxy, 'Yxy', 'srgb' )

    JOD, m_stats = cvvdp.predict( I_test[kk], I_ref, dim_order="HWC" )
    Q_JOD[kk] = JOD.item()

    PSNR, m_stats = psnr_met.predict( I_test[kk], I_ref, dim_order="HWC" )
    Q_PSNR[kk] = PSNR.item()

    q_str = f'Chroma subsampling {ss_factors[kk]} - Quality: {Q_JOD[kk]:.3f} JOD; {Q_PSNR[kk]:.3f} db'
    print( q_str )


f, axs = plt.subplots(2, 3)
axs[0][0].plot( ss_factors, Q_JOD, '-or' )
axs[0][0].set_ylabel('ColourVideoVDP [JOD]', color='r') 
axs[0][0].tick_params(axis='y', labelcolor='r')
axs[0][0].set_xscale("log")
axs[0][0].set_xticks(ss_factors)
axs[0][0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axs[0][0].set_xlabel( 'Subsample factor' )
axs[0][0].grid(True)

ax2 = axs[0][0].twinx()
ax2.plot( ss_factors, Q_PSNR, '-sb' )
ax2.set_ylabel('RGB-PSNR [dB]', color='b') 
ax2.tick_params(axis='y', labelcolor='b')

patch_sz = 256 # Show only a portion of the image
for kk in range(len(ss_factors)):
    row = (kk+1) % 3
    col = int( (kk+1)/3 )
    axs[col][row].imshow( I_test[kk][-patch_sz:,-patch_sz:,:] )
    axs[col][row].set_xticks([])
    axs[col][row].set_yticks([])
    axs[col][row].set_title(f'Chroma ss factor {ss_factors[kk]}')

f.show()
plt.waitforbuttonpress()

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
