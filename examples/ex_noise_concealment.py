# This example creates a band-limited noise pattern and "conceals" it in the
# image. The amplitudes of the noise are optimized so that the PSNR for the
# optimized image is the same as for the initial image but the visibility of the
# noise is minimised (in terms of ColorVideoVDP). As the result of the
# optimization, the noise is reallocated to darker and textured areas and
# initial band-limited noise is turned into high frequency (blue) noise.
# 
# The example shows how to use ColorVideoVDP as a perceptual optimization
# criterion.


# Important: This and other examples should be executed from the main
# ColorVideoVDP directory: python examples/ex_<...>.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import ex_utils as utils
import pycvvdp
import time
import imageio.v2 as io

from torchvision.transforms import GaussianBlur

debug = False
save_results = False
if save_results:
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

class NoiseField(torch.nn.Module):
    def __init__(self, bkg_img):
        super().__init__()    
        self.bkg_img = bkg_img
        sh = bkg_img.shape
        sh_nf = list(sh)
        sh_nf[-3] = 1 # Set channels to 1

        # The amplitudes are in the log space to avoid negative values
        self.amplitudes = torch.nn.Parameter( torch.zeros( sh_nf, device=bkg_img.device ) ) 

        # Band-limited noise
        sigma1 = 3
        kernel_size = 2 * int(4 * sigma1 + 0.5) + 1
        blur1 = GaussianBlur(kernel_size=kernel_size, sigma=sigma1)
        sigma2 = 1
        kernel_size = 2 * int(4 * sigma2 + 0.5) + 1
        blur2 = GaussianBlur(kernel_size=kernel_size, sigma=sigma2)
        noise = (torch.rand_like( bkg_img )*0.8).clamp(0,1)
        self.noise = blur2(noise) - blur1(noise)
        self.metric = pycvvdp.cvvdp(display_name='standard_4k')

        with torch.no_grad():    
            self.target_mse = ((self()-self.bkg_img)**2).mean()

    def forward(self):
        return (self.bkg_img + (2**self.amplitudes)*self.noise).clamp(0,1)

    def loss(self, pred, y) -> torch.Tensor:
        alpha = 100000
        # return self.metric.loss( pred, y, dim_order="CHW") + alpha * ((2**self.amplitudes).sum()/self.amplitudes.numel()-1)**2
        return self.metric.loss( pred, y, dim_order="CHW") + alpha * (((pred-y)**2).mean()-self.target_mse)**2
    
    def psnr(self):
        mse = torch.mean((self.forward() - self.bkg_img) ** 2)
        return -10 * torch.log10(mse)        


device = torch.device("cuda:0")

I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
# The shape of the image will be CHW
T_ref = torch.as_tensor( I_ref.astype(np.float32) ).to(device).permute((2,0,1))/np.iinfo( I_ref.dtype ).max

model = NoiseField( T_ref )

model.to(device)

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0, weight_decay=0, dampening=0)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.5, eps=1e-3, amsgrad=True )
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2 )

loss_fn = lambda pred, y : model.loss( pred, y )

I_initial = model().detach().permute((1,2,0)).cpu().numpy()
with torch.no_grad():    
    PSNR_initial = model.psnr()


plt.ion()
fig = plt.figure(figsize=(20, 10))
ax = [None]*4
ax[0] = plt.subplot2grid((3, 3), (0, 0), rowspan=2)
ax[1] = plt.subplot2grid((3, 3), (0, 1), rowspan=2)
ax[2] = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
ax[3] = plt.subplot2grid((3, 3), (2, 0), colspan=3)
ax_gm = ax[3].twinx()  # Second y-axis

grad_mag = -1

max_iter = 1001

loss_tab = np.ones( (max_iter), dtype=np.float32 ) * np.nan
grad_mag_tab = np.ones( (max_iter), dtype=np.float32 ) * np.nan

for kk in range(max_iter):
    print( f"Iteration {kk}" )
    optimizer.zero_grad()

    pred = model()
    loss = loss_fn(pred, T_ref) 

    loss_tab[kk] = loss.item()

    if kk % 20 == 0:
           
        opt_img = pred.detach().permute((1,2,0)).cpu().numpy()
        ax[0].clear()
        ax[0].imshow( (I_initial*255).astype(np.uint8) )
        ax[0].set_title( f"Initial image ({PSNR_initial} dB)" )
        ax[1].clear()
        ampl = (2**model.amplitudes).detach().permute((1,2,0)).cpu().numpy()
        ampl_norm = ampl.sum()/ampl.size
        ax[1].imshow( (ampl/2*255).astype(np.uint8) )
        ax[1].set_title( f"Amplitudes {ampl_norm}" )
        with torch.no_grad():    
            PSNR_current = model.psnr()
        ax[2].clear()
        ax[2].imshow( opt_img )
        ax[2].set_title( f"Optimized ({PSNR_current} dB)" )

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
        
        for pp in range(2):
            ax[pp].set_xticks([])        
            ax[pp].set_yticks([])        

        plt.tight_layout()

        if save_results and kk % 100 == 0:
            io.imwrite( f'{output_dir}/reconstructed_image_i{kk:04d}.png', (opt_img*255).astype(np.ubyte) )

        fig.canvas.draw()
        fig.canvas.flush_events()
 
    #time.sleep(0.1)

    # Backpropagation
    loss.backward()

    grad_mag_tab[kk] = model.amplitudes.grad.norm(p=2)*255

    optimizer.step()

plt.waitforbuttonpress()