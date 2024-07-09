# This example tests ColorVideoVDP performance as a loss function. It is inspired by the analysis presented in Sec. 3 of https://doi.org/10.1007/s11263-020-01419-7
#
# The code will optimize for pixel values in an image so that they match the pixel values in a reference image (no network, direct reconstruction). 
# The optimization will success in reconstructing the reference image if the initialization is sufficiently close to the reference images. A random 
# initialization will cause the optimization to get stuck in a local minimum. 

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py


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
    return torch.diff(Y[1,:,:],dim=-1).abs().sum() + torch.diff(Y[1,:,:],dim=-2).abs().sum() + torch.diff(Y[2,:,:],dim=-1).abs().sum() + torch.diff(Y[2,:,:],dim=-2).abs().sum()

I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
# patch_sz=256
# I_ref = I_ref[-patch_sz:,-patch_sz:,:]

T_ref = torch.as_tensor( I_ref.astype(np.float32) ).to(device).permute((2,0,1))/(2**16-1)

model = ImageRecovery( T_ref, initialization="blurred" )
#model = ImageRecovery( T_ref, initialization="random" )

model.to(device)

#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0, weight_decay=0, dampening=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

cvvdp = pycvvdp.cvvdp(display_name='standard_4k')

# Use a "pure" cvvdp loss only if the initialization is close to the reference image (e.g., "blurred" passed to ImageRecovery above)
loss_fn = lambda pred, y : cvvdp.loss( pred, y, dim_order="CHW")

# A Mixture of cvvdp loss and L2 works better with random initialization
# loss_fn = lambda pred, y : cvvdp.loss( pred, y, dim_order="CHW") + 100*torch.mean((pred - y)**2)

plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

for kk in range(1001):
    print( f"Iteration {kk}" )
    optimizer.zero_grad()
    pred = model().clamp(0.,1.)
    loss = loss_fn(pred, T_ref)

    if kk % 20 == 0:
           
        opt_img = pred.detach().permute((1,2,0)).cpu().numpy()
        ax[0].imshow( opt_img )
        ax[0].set_title( "Optimized" )
        ax[1].imshow( (I_ref/256).astype(np.uint8) )
        ax[1].set_title( "Target" )

        fig.suptitle( f"Iteration {kk}: loss {loss.item()}" )
        
        for pp in range(2):
            ax[pp].set_xticks([])        
            ax[pp].set_yticks([])        

        # plt.tight_layout()

        if save_results and kk % 100 == 0:
            io.imwrite( f'reconstructed_image_i{kk:04d}.png', (opt_img*255).astype(np.ubyte) )

        fig.canvas.draw()
        fig.canvas.flush_events()
 
    #time.sleep(0.1)

    # Backpropagation
    loss.backward()
    optimizer.step()

plt.waitforbuttonpress()