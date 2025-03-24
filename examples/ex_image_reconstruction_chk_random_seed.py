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
import random
from torchvision.transforms import GaussianBlur

## Torch Gaussian blur to handle the phase uncertainty in cvvdp 
## causes non-deterministic behavior in the optimization with 'Adam' in 'cuda' environment
## This is due to the 'reflect' padding in the Gaussian blur in torchvision.transforms
## The issue is resolved by using the custom Gaussian blur with 'replicate' padding in the cvvdp library

config_path_debug = ['./pycvvdp/vvdp_data_debug']
config_path_original= ['./pycvvdp/vvdp_data']   

padding_type = 'custom_gaussian_blur' # 'torch_gaussian_blur' or 'custom_gaussian_blur'
if padding_type == 'torch_gaussian_blur':
    vvdp_config_path = config_path_original
else:
    vvdp_config_path = config_path_debug

device = torch.device("cuda:0") # "cuda:0" or "cpu"
seed_type = 'identical_seed' # 'identical_seed' or 'different_seed'
optimizer_type = 'adam' # 'sgd' or 'adam'
num_seed= 2
num_iters = 101

if device.type == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
    # torch.use_deterministic_algorithms(True)
elif device.type == "cuda":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.autograd.set_detect_anomaly(True)

debug = False
save_results = False

class ImageRecovery(torch.nn.Module):
    def __init__(self, ref_img, initialization="random", seed=None):
        super().__init__()        
        if initialization == "random":
            if seed is not None:
                torch.manual_seed(seed)
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
T = torch.zeros(2, *T_ref.shape).to(device)

if seed_type=='identical_seed':
    seed_vec = [0, 0]
else:
    seed_vec = [0, 1]

for n in range(num_seed):
    seed = seed_vec[n]
    torch.manual_seed(seed)
    T[n,...] = torch.rand_like(T_ref).to(device)

cvvdp = pycvvdp.cvvdp(display_name='standard_4k', config_paths=vvdp_config_path)
loss_fn = lambda pred, y : cvvdp.loss( pred, y, dim_order="CHW")

if torch.equal(T[0, ...], T[1, ...]):
    print("T[0,...] is equal to T[1,...]")
else:
    print("T[0,...] is not equal to T[1,...]")

plt.ion()
fig, ax = plt.subplots(num_seed,3, figsize=(16, 8))
loss_1 =0
loss_2 =0
track_lr = torch.zeros((num_seed, 101))
track_loss = torch.zeros((num_seed, 101))
T_result = torch.zeros_like(T)
for n in range(num_seed):
    x = T[n,...].requires_grad_(True)
    optvars = [{'params': x}]
    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(optvars, lr=0.01)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(optvars, lr=0.01)
    for kk in range(num_iters):
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    state = optimizer.state[param]
                    if 'exp_avg_sq' in state:  # Adam keeps track of this
                        lr_t = param_group['lr'] / (state['exp_avg_sq'].sqrt() + 1e-8)
                        print(f"Iteration {kk+1}: Effective LR for param {param.shape}: {lr_t.mean().item()}")
                        track_lr[n,kk]=lr_t.mean().item()

        optimizer.zero_grad()
        pred = x.clamp(0., 1.)
        loss = loss_fn(pred, T_ref)
        track_loss[n,kk] = loss.detach().item()    

        if kk==0:
            ax[0, 1].imshow((I_ref / 256).astype(np.uint8))
            ax[0, 1].set_title("Target")

        if kk % 20 == 0:
            opt_img = pred.detach().permute((1, 2, 0)).cpu().numpy()
            ax[n, 0].imshow(opt_img)
            ax[n, 0].set_title(f"Optimized {n}")


            if n==0:
                loss_1 = loss.item()
            else:
                loss_2 = loss.item()

            fig.suptitle(f"Iteration {kk}, device: {device.type}, opt: {optimizer_type}, seed: {seed_vec}")

            for pp in range(2):
                ax[n, pp].set_xticks([])
                ax[n, pp].set_yticks([])

        if kk == num_iters-1:
            T_result[n,...] = pred

        fig.canvas.draw()
        fig.canvas.flush_events()
        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, 
        #                                torch.profiler.ProfilerActivity.CUDA], 
        #                     record_shapes=True) as prof:
        #     # Backpropagation
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        loss.backward()
        optimizer.step()

diff_img = torch.abs(T_result[0,...] - T_result[1,...]).squeeze().detach().permute((1,2,0)).cpu().numpy()
diff_img = (100*diff_img / 256).clip(0, 1)
ax[1, 1].imshow((diff_img).astype(np.uint8))
ax[1, 1].set_title("Image Difference X 100")
ax[0, 2].plot(np.abs(track_loss[0, :].cpu().numpy()-track_loss[1, :].cpu().numpy()))
ax[0, 2].set_title("Loss difference")
ax[1, 2].plot(np.abs(track_lr[0, :].detach().cpu().numpy()-track_lr[1, :].detach().cpu().numpy()))
ax[1, 2].set_title("LR difference")

import pdb
pdb.set_trace()
plt.waitforbuttonpress()
