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
import copy

from torchmetrics import StructuralSimilarityIndexMeasure

import cv2

# For debugging only
# from gfxdisp.pfs.pfs import pfs

# This is a 16-bit image, convert to float
I_ref_np16 = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))

patch_sz = 256 # Use only a portion of the image
I_ref = I_ref_np16[-patch_sz:,-patch_sz:,:].astype(np.float32) / np.iinfo(I_ref_np16.dtype).max

# I_ref = reshuffle_dims( torch.as_tensor(I_ref_np), "HWC", "CFHW" )
# ct = ColorTransform()
# I_DKL = ct.rgb2colourspace(I_ref, "DKLd65")

# sigma = 4
# GB = tt.GaussianBlur(math.ceil(sigma*3)*2+1, sigma)
# for cc in range(1, 3):
#     I_DKL[cc,:,:,:] = GB(I_DKL[cc,:,:,:])

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
            I_ss = cv2.resize(I_ref, dsize=dim_ss, interpolation=cv2.INTER_CUBIC)
            I_test[tt][kk] = cv2.resize(I_ss, dsize=dim, interpolation=cv2.INTER_CUBIC)
        elif subsampling == 'Chroma-ss Yxy': # chroma subsampling, Yxy
            chroma_ss = cv2.resize(I_Yxy[:,:,1:3], dsize=dim_ss, interpolation=cv2.INTER_CUBIC)
            chroma_rec = cv2.resize(chroma_ss, dsize=dim, interpolation=cv2.INTER_CUBIC)
            I_Yxy[:,:,1:3] = chroma_rec
            I_test[tt][kk] = utils.im_ctrans( I_Yxy, 'Yxy', 'srgb' ).clip(0.,1.)
        elif subsampling == 'Chroma-ss YCbCr': # chroma subsampling, YCbCr
            chroma_ss = cv2.resize(I_YCbCr[:,:,1:3], dsize=dim_ss, interpolation=cv2.INTER_CUBIC)
            chroma_rec = cv2.resize(chroma_ss, dsize=dim, interpolation=cv2.INTER_CUBIC)
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



M = len(Q[0]) # The number of metrics

f, axs = plt.subplots(len(ss_type), N+M, layout="constrained")
if len(ss_type) == 1:
    axs = [axs]

qpp = { 'cvvdp': { 'ylabel': 'ColourVideoVDP JOD', 'ylim': (5, 10) }, 
        'ssim-lum': { 'ylabel': 'SSIM-luma', 'ylim': (0, 1) },
        'ssim-rgb': { 'ylabel': 'SSIM-RGB', 'ylim': (0, 1) } }

for tt in range(len(ss_type)):
    for mm in range(M):
        qm_key = list(Q[tt].keys())[mm]
        axs[tt][mm].plot( ss_factors, Q[tt][qm_key], '-or' )
        axs[tt][mm].set_ylabel( qpp[qm_key]['ylabel'] ) 
        axs[tt][mm].set_ylim( qpp[qm_key]['ylim'] ) 
        #axs[tt][mm].tick_params(axis='y', labelcolor='r')
        axs[tt][mm].set_xscale("log")
        axs[tt][mm].set_xticks(ss_factors)
        axs[tt][mm].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axs[tt][mm].set_xlabel( 'Subsample factor' )
        axs[tt][mm].grid(True)

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

f.show()
plt.waitforbuttonpress()
# plt.savefig('chroma-ss.png')

