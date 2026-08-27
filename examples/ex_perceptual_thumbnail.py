# This example shows how to create a thumbnail of an image that is possibly
# similar to the full resolution image. The idea is inspired by the paper:
#
# Perceptually based downscaling of images
# AC Oztireli, M Gross
# ACM Transactions on Graphics (TOG) 34 (4), 1-10
# https://doi.org/10.1145/2766891
#
# Given an input image, the code generates an image that with the resolution
# reduced by a factor of R (8 by default). The optimization loop finds values of
# all pixels, so that when the thumbnail image is upsampled with
# nearest-neighbor to the resolution of the input image, the difference between
# the two is the smallest in terms of ColorVideoVDP loss.

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pycvvdp
import imageio.v2 as io

debug = False
save_results = False
output_dir = "perceptual_thumbnail"
if save_results:
    os.makedirs(output_dir, exist_ok=True)

R = 6  # Downsampling factor

class ThumbnailModel(torch.nn.Module):
    def __init__(self, ref_img, R=8):
        super().__init__()
        self.R = R
        # Initialize the thumbnail as a box/average downsample of the reference
        self.init_thumb = torch.nn.functional.avg_pool2d(ref_img.unsqueeze(0), kernel_size=R).squeeze(0)
        self.thumb = torch.nn.Parameter(self.init_thumb.clone())

    def forward(self):
        # Nearest-neighbor upsample the thumbnail back to full resolution
        return torch.nn.functional.interpolate(self.thumb.unsqueeze(0), scale_factor=self.R, mode="nearest").squeeze(0)


device = torch.device("cuda:0")

#I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'palm_beach.png'))
I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'perc_downscaling_face_1.png'))


# Crop the image so that its dimensions are divisible by R
H, W = I_ref.shape[:2]
H_crop = (H // R) * R
W_crop = (W // R) * R
I_ref = I_ref[:H_crop, :W_crop, :]

T_ref = torch.as_tensor( I_ref.astype(np.float32) ).to(device).permute((2,0,1))/np.iinfo( I_ref.dtype ).max

model = ThumbnailModel( T_ref, R=R )

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2 )

cvvdp = pycvvdp.cvvdp(display_name='standard_4k' )
# cvvdp = pycvvdp.cvvdp(display_name='standard_4k', config_paths=['../metric_configs/cvvdp_add_mutual/cvvdp_parameters.json'] )

loss_fn = lambda pred, y : cvvdp.loss( pred, y, dim_order="CHW")

plt.ion()
fig = plt.figure(figsize=(16, 8))
ax = [None]*4
ax[0] = plt.subplot2grid((2, 3), (0, 0))
ax[1] = plt.subplot2grid((2, 3), (0, 1))
ax[2] = plt.subplot2grid((2, 3), (0, 2))
ax[3] = plt.subplot2grid((2, 3), (1, 0), colspan=3)
ax_gm = ax[3].twinx()  # Second y-axis

grad_mag = -1

max_iter = 1001

loss_tab = np.ones( (max_iter), dtype=np.float32 ) * np.nan
grad_mag_tab = np.ones( (max_iter), dtype=np.float32 ) * np.nan

thumb_naive_t = torch.nn.functional.interpolate(model.init_thumb.unsqueeze(0), scale_factor=R, mode="nearest").squeeze(0)        
th, tw = model.thumb.shape[-2], model.thumb.shape[-1]
thumb_naive_t[...,-th:,-tw:] = model.init_thumb
thumb_naive = thumb_naive_t.detach().clamp(0, 1).permute((1,2,0)).cpu().numpy()


for kk in range(max_iter):
    print( f"Iteration {kk}" )
    optimizer.zero_grad()

    pred = model()
    loss = loss_fn(pred.clamp(0, 1), T_ref)  # clamp only in loss call

    loss_tab[kk] = loss.item()

    if kk % 20 == 0:

        # thumb_img = model.thumb.detach().clamp(0, 1).permute((1,2,0)).cpu().numpy()
        opt_img = pred.detach().clamp(0, 1).permute((1,2,0)).cpu().numpy()

        ax[0].clear()
        ax[0].imshow( thumb_naive )
        ax[0].set_title( f"Naive thumbnail ({th}x{tw})" )
        ax[1].clear()
        disp_img = torch.nn.functional.interpolate(model.thumb.detach().unsqueeze(0), scale_factor=R, mode="nearest").squeeze(0)        
        disp_img[...,-th:,-tw:] = model.thumb.detach()
        ax[1].imshow( disp_img.detach().clamp(0, 1).permute((1,2,0)).cpu().numpy() )
        ax[1].set_title( "Perceptual thumbnail" )
        ax[2].clear()
        ax[2].imshow( I_ref.astype(np.float32)/np.iinfo(I_ref.dtype).max )
        ax[2].set_title( "Reference" )

        ax[3].clear()
        it_x = range(max_iter)
        ax[3].plot( it_x, loss_tab )
        ax[3].set_xlabel( 'Iteration' )
        ax[3].set_ylabel( 'Loss [JOD]' )
        ax[3].set_yscale( 'log' )

        ax_gm.clear()
        color2 = 'tab:red'
        ax_gm.plot( it_x, grad_mag_tab, color=color2, label='Gradient magnitude')
        ax_gm.yaxis.set_label_position("right")
        ax_gm.set_ylabel('Gradient magnitude', color=color2)
        ax_gm.tick_params(axis='y', labelcolor=color2)

        fig.suptitle( f"Iteration {kk}: loss {loss.item():.4f}" )

        for pp in range(3):
            ax[pp].set_xticks([])
            ax[pp].set_yticks([])

        plt.tight_layout()

        if save_results and kk % 100 == 0:
            io.imwrite( f'{output_dir}/upsampled_preview_i{kk:04d}.png', (opt_img*255).astype(np.ubyte) )

        fig.canvas.draw()
        fig.canvas.flush_events()

    # Backpropagation
    loss.backward()

    grad_mag_tab[kk] = model.thumb.grad.norm(p=2)*255

    optimizer.step()

    with torch.no_grad():
        model.thumb.clamp_(0., 1.)  # clamp param after step

if save_results:
    thumb_img = model.thumb.detach().clamp(0, 1).permute((1,2,0)).cpu().numpy()
    opt_img = model().detach().clamp(0, 1).permute((1,2,0)).cpu().numpy()
    io.imwrite( f'{output_dir}/thumbnail.png', (thumb_img*255).astype(np.ubyte) )
    io.imwrite( f'{output_dir}/upsampled_preview.png', (opt_img*255).astype(np.ubyte) )

plt.waitforbuttonpress()
