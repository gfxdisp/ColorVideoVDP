# This is an example showing how ColorVideoVDP can be used as a loss function to adaptively reduce chromatic details in YCbCr color space. 
# It will reproduce example from Fig. 20 in ColorVideoVDP paper (https://doi.org/10.1145/3658144). 
# See Section 6.1 in that paper for the full explanation. 

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import ex_utils as utils
import pycvvdp
from pycvvdp.video_source import reshuffle_dims
import torchvision.transforms as tt
import math
import copy

from torchmetrics import StructuralSimilarityIndexMeasure

import cv2
#import PIL

def resize_array(img, dsize):
    #return np.array( PIL.Image.fromarray(img).resize(dsize, resample=PIL.Image.LANCZOS) )
    return cv2.resize( img, dsize=dsize, interpolation=cv2.INTER_LANCZOS4 )

# For debugging only
# from gfxdisp.pfs.pfs import pfs

I_ref_np16 = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))

patch_sz = 256 # Use only a portion of the image
# This is a 16-bit image, convert to float
I_ref = I_ref_np16[-patch_sz:,-patch_sz:,:].astype(np.float32) / np.iinfo(I_ref_np16.dtype).max

epsilon = 1e-4 # to avoid div by 0
I_Yxy = utils.im_ctrans( utils.im_ctrans( I_ref, 'srgb', 'rgb709' ) + epsilon, 'rgb709', 'Yxy' )
I_YCbCr = utils.srgb2ycbcr(I_ref)

cvvdp = pycvvdp.cvvdp(display_name='standard_4k')
psnr_met = pycvvdp.pu_psnr_rgb2020()
psnr_met.set_display_model(display_name='standard_4k')

ss_factors = [1.5, 2, 4, 8, 16]
N = len(ss_factors)

met_dict = { 'cvvdp': [], 'ssim-lum': [], 'ssim-rgb': [] }
# met_dict = { 'cvvdp': [], 'ssim-rgb': [] }
Q = [ copy.deepcopy(met_dict) for _ in range(len(met_dict)) ] 
I_test = [[None] * N, [None] * N, [None] * N]

ssim = StructuralSimilarityIndexMeasure(data_range=1)

ss_type = ["RGB-ss", "Chroma-ss Yxy", "Chroma-ss YCbCr"]
# ss_type = ["Chroma-ss YCbCr"]

for tt, subsampling in enumerate(ss_type): # For each type of subsampling
    for kk in range(len(ss_factors)):

        dim = (I_ref.shape[1], I_ref.shape[0])
        dim_ss = (int(I_ref.shape[1]/ss_factors[kk]), int(I_ref.shape[0]/ss_factors[kk]))

        if subsampling == 'RGB-ss': # regular subsampling
            I_ss = resize_array(I_ref, dsize=dim_ss)
            I_test[tt][kk] = resize_array(I_ss, dsize=dim)
        elif subsampling == 'Chroma-ss Yxy': # chroma subsampling, Yxy
            chroma_ss = resize_array(I_Yxy[:,:,1:3], dsize=dim_ss)
            chroma_rec = resize_array(chroma_ss, dsize=dim)
            I_Yxy[:,:,1:3] = chroma_rec
            I_test[tt][kk] = utils.im_ctrans( I_Yxy, 'Yxy', 'srgb' ).clip(0.,1.)
        elif subsampling == 'Chroma-ss YCbCr': # chroma subsampling, YCbCr
            chroma_ss = resize_array(I_YCbCr[:,:,1:3], dsize=dim_ss)
            chroma_rec = resize_array(chroma_ss, dsize=dim)
            I_YCbCr[:,:,1:3] = chroma_rec
            I_test[tt][kk] = utils.ycbcr2srgb( I_YCbCr ).clip(0.,1.)

        JOD, m_stats = cvvdp.predict( I_test[tt][kk], I_ref, dim_order="HWC" )
        Q[tt]["cvvdp"].append(JOD.item())

        #PSNR, m_stats = psnr_met.predict( I_test[kk], I_ref, dim_order="HWC" )
        #Q_PSNR[kk] = PSNR.item()


        if 'ssim-lum' in Q[tt]:
            Y_test = utils.srgb2ycbcr(I_test[tt][kk])[:,:,0:1]
            Y_ref = utils.srgb2ycbcr(I_ref)[:,:,0:1]
            ssqi = ssim(reshuffle_dims(torch.tensor(Y_test), in_dims="HWC", out_dims="BCHW"), reshuffle_dims(torch.tensor(Y_ref), in_dims="HWC", out_dims="BCHW"))
            Q[tt]["ssim-lum"].append(ssqi.item())

        if 'ssim-rgb' in Q[tt]:
            ssqi = ssim(reshuffle_dims(torch.tensor(I_test[tt][kk]), in_dims="HWC", out_dims="BCHW"), reshuffle_dims(torch.tensor(I_ref), in_dims="HWC", out_dims="BCHW"))
            Q[tt]["ssim-rgb"].append(ssqi.item())

        #q_str = f'Chroma subsampling {ss_factors[kk]} - Quality: {Q_JOD[kk]:.3f} JOD; {Q_PSNR[kk]:.3f} db'
        #print( q_str )


mplots = [ {'ylabel': 'Quality [JOD]', 'ylim': (5.5, 10), 'metrics': ['cvvdp'] }, 
           {'ylabel': 'SSIM', 'ylim': (0.4, 1), 'metrics': ['ssim-lum', 'ssim-rgb'] } ]

M = len(mplots) #len(Q[0]) # The number of metrics

fig, axs = plt.subplots(len(ss_type), N+M, layout="constrained", figsize=(18, 8) )
if len(ss_type) == 1:
    axs = [axs]

# qpp = { 'cvvdp': { 'ylabel': 'ColourVideoVDP JOD', 'ylim': (4, 10), 'col': 0 }, 
#         'ssim-lum': { 'ylabel': 'SSIM-luma', 'ylim': (0.4, 1), 'col': 1 },
#         'ssim-rgb': { 'ylabel': 'SSIM-RGB', 'ylim': (0.4, 1), 'col': 1 } }

nice_name = { 'cvvdp': 'ColorVideoVDP', 'ssim-rgb': 'SSIM (RGB)', 'ssim-lum': 'SSIM (luma)'}

for tt in range(len(ss_type)):
    first_row = tt==0
    last_row = tt == (len(ss_type)-1)
    for mm in range(M):
        qm_key = list(Q[tt].keys())[mm]
        for qm in mplots[mm]["metrics"]:
            axs[tt][mm].plot( ss_factors, Q[tt][qm], '-o', label=nice_name[qm] )
        axs[tt][mm].set_ylim( mplots[mm]['ylim'] ) 
        axs[tt][mm].set_xscale("log")
        axs[tt][mm].set_xticks(ss_factors)
        axs[tt][mm].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())        
        if last_row:
            axs[tt][mm].set_xlabel( 'Subsample factor' )
        if first_row:
            axs[tt][mm].legend()
        axs[tt][mm].grid(True)
        axs[tt][mm].set_ylabel( mplots[mm]['ylabel'] ) 
            #axs[tt][mm].tick_params(axis='y', labelcolor='r')

    # ax2 = axs[0][0].twinx()
    # ax2.plot( ss_factors, Q_PSNR, '-sb' )
    # ax2.set_ylabel('RGB-PSNR [dB]', color='b') 
    # ax2.tick_params(axis='y', labelcolor='b')

    for kk in range(N):
        col = kk+M
        axs[tt][col].imshow( I_test[tt][kk] )
        axs[tt][col].set_xticks([])
        axs[tt][col].set_yticks([])
        axs[tt][col].set_title(f'{ss_type[tt]} x {ss_factors[kk]}')

plt.savefig( 'chroma-ss.pdf', bbox_inches='tight' )  

fig.show()
plt.waitforbuttonpress()
# plt.savefig('chroma-ss.png')

