# This is an example showing how ColorVideoVDP can be used as a loss function to adaptively reduce chromatic details in YCbCr color space. 
# It will reproduce example from Fig. 21 in ColorVideoVDP paper (https://doi.org/10.1145/3658144). 
# See Section 6.3 in that paper for the full explanation. 

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pycvvdp
import imageio.v2 as io

from torchvision.transforms import GaussianBlur

debug = False
save_results = False

class ImageRecovery(torch.nn.Module):
    def __init__(self, ref_img, initialization="random"):
        super().__init__()        
        if initialization == "random":
            self.rec_image = torch.nn.Parameter( torch.rand_like( ref_img ) )
        elif initialization == "blurred":
            sigma = 4
            gb = GaussianBlur(int(sigma*4)+1, sigma)
            self.rec_image = torch.nn.Parameter(gb.forward(ref_img))
        elif initialization == "noise":
            self.rec_image = torch.nn.Parameter((ref_img + torch.rand_like( ref_img )*0.2).clamp(0,1))
        elif initialization == "same":
            self.rec_image = torch.nn.Parameter(ref_img.clone())
        else:
            assert False
        
    def forward(self):
        return self.rec_image


device = torch.device("cuda:0")

_rgb_rec7092ycbcr = torch.as_tensor([[0.298999944347618, 0.587000125991912, 0.113999929660470],\
  [-0.168735860241319,  -0.331264179453675,   0.500000039694994],\
  [0.500000039694994,  -0.418687679024188,  -0.081312360670806]], device=device)


def srgb2ycbcr( X ):    
    Y = torch.empty_like(X)
    M = _rgb_rec7092ycbcr
    for cc in range(3):
        Y[cc,:,:] = torch.sum(X*(M[cc,:].view(3,1,1)), dim=-3, keepdim=True)
    return Y

def reduce_chroma( I ):
    Y = srgb2ycbcr(I)
    return torch.diff(Y[1,:,:],dim=-1).abs().mean() + torch.diff(Y[1,:,:],dim=-2).abs().mean() + torch.diff(Y[2,:,:],dim=-1).abs().mean() + torch.diff(Y[2,:,:],dim=-2).abs().mean()

I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
# patch_sz=256
# I_ref = I_ref[-patch_sz:,-patch_sz:,:]

T_ref = torch.as_tensor( I_ref.astype(np.float32) ).to(device).permute((2,0,1))/(2**16-1)

model = ImageRecovery( T_ref, initialization="same" )
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

cvvdp = pycvvdp.cvvdp(display_name='standard_4k')

loss_fn = lambda pred, y : cvvdp.loss( pred, y, dim_order="CHW") + 100*reduce_chroma(pred)

plt.ion()
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

for kk in range(1001):
    print( f"Iteration {kk}" )
    optimizer.zero_grad()
    pred = model().clamp(0.,1.)
    loss = loss_fn(pred, T_ref)

    if kk % 20 == 0:
        Ycbcr = srgb2ycbcr(pred.detach())
        if kk==0:
            rng_cb = (Ycbcr[1,:,:].min().item(), Ycbcr[1,:,:].max().item())
            rng_cr = (Ycbcr[2,:,:].min().item(), Ycbcr[2,:,:].max().item())
            
        opt_img = pred.detach().permute((1,2,0)).cpu().numpy()
        ax[0].imshow( opt_img )
        ax[1].imshow( Ycbcr[1,:,:].cpu().numpy(), vmin=rng_cb[0], vmax=rng_cb[1] )
        ax[1].set_title( 'Cb' )
        ax[2].imshow( Ycbcr[2,:,:].cpu().numpy(), vmin=rng_cr[0], vmax=rng_cr[1] )
        ax[2].set_title( 'Cr' )

        fig.suptitle( f"Iteration {kk}: loss {loss.item()}" )
        
        for pp in range(3):
            ax[pp].set_xticks([])        
            ax[pp].set_yticks([])        

        plt.tight_layout()

        if save_results and kk % 20 == 0:
            plt.savefig( f'adaptive_chroma_channels_i{kk:04d}.png' )
            #io.imwrite( f'adaptive_chroma_image_i{kk:04d}.png', (opt_img*255).astype(np.ubyte) )

        fig.canvas.draw()
        fig.canvas.flush_events()
 
    # Backpropagation
    loss.backward()
    optimizer.step()

plt.waitforbuttonpress()
