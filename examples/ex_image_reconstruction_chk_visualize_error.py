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
import time, math
import imageio
import random
from torchvision.transforms import GaussianBlur
# import pytorch_warmup as warmup

device = torch.device("cuda:0") # "cuda:0" or "cpu"
seed_type = 'identical_seed' # 'identical_seed' or 'different_seed'
optimizer_type = 'adam' # 'sgd' or 'adam'
num_seed = 1
num_iters = 1500
noise_scale = 1.0
loss_type = 'cvvdp'
lr = 0.01
WRITE_FREQ = 10
simulated_annealing_schedule = np.linspace(0.05, 0.0, num_iters)
batch_size_schedule = [max(1, int(np.round((200*(annealing**2))/(0.05**2)))) for annealing in simulated_annealing_schedule]
print(simulated_annealing_schedule[::100])
print(batch_size_schedule[::100])

config_path_debug = ['./pycvvdp/metric_configs/cvvdp_mult_mutual_diff_sensitivity']
config_name = config_path_debug[0].split('/')[-1]
vvdp_config_path = config_path_debug

output_path = f'examples/config_{config_name}_optimization_noise{noise_scale}_iter{num_iters}_lossCVVDPSimulatedAnnealing{simulated_annealing_schedule[0]}to{simulated_annealing_schedule[-1]}_lr{lr}timesbatch_batch{batch_size_schedule[0]}to{batch_size_schedule[-1]}.mp4'
print('Writing to ', output_path)
warmup_iters = 0

if device.type == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
    # torch.use_deterministic_algorithms(True)
elif device.type == "cuda":
    # Solution 1. Set the environment variable CUBLAS_WORKSPACE_CONFIG to :4096:8 (Failed)
    # export CUBLAS_WORKSPACE_CONFIG=:4096:8
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)
    # torch.autograd.set_detect_anomaly(True)

    # Solution 2. Change the Gaussian filter padding type 
    # Implemented in GaussianBlur function in torchvision.transoforms (Worked)
    # But, the parameter of cvvdp is fitted with the default padding type.

debug = False
save_results = False

# Helper function to convert channel-first images to channel-last if needed
def to_imarray(img):
    img_np = img.detach().cpu().numpy() if hasattr(img, 'detach') else np.array(img)
    if img_np.ndim == 3 and img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    return img_np

def init_image(ref_img, initialization="random", noise_scale=0.2, seed=None):
    if initialization == "random":
        if seed is not None:
            torch.manual_seed(seed)
        rec_image = torch.nn.Parameter( torch.rand_like( ref_img ) )
    elif initialization == "blurred":
        sigma = 4
        gb = GaussianBlur(int(sigma*4)+1, sigma)
        rec_image = torch.nn.Parameter(gb.forward(ref_img))
    elif initialization == "noise":
        rec_image = torch.nn.Parameter((ref_img + (-0.5+torch.rand_like( ref_img ))*noise_scale).clamp(0,1))
    elif initialization == "same":
        rec_image = torch.nn.Parameter(ref_img.clone())
    else:
        assert False
    
    return rec_image

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

cvvdp = pycvvdp.cvvdp(display_name='standard_4k', config_paths=vvdp_config_path)
# cvvdp = pycvvdp.cvvdp(display_name='standard_4k')
if loss_type == 'cvvdp':
    loss_fn = lambda pred, y : cvvdp.loss( pred, y, dim_order="CHW")
else:
    loss_fn = torch.nn.MSELoss()
mse_loss = torch.nn.MSELoss()

T_ref = torch.as_tensor( I_ref.astype(np.float32) ).to(device).permute((2,0,1))/(2**16-1)
T_ref_np = to_imarray(T_ref)
print(T_ref.min(), T_ref.max())
# Original code (unchanged)
x = init_image(T_ref, initialization="noise" if noise_scale < 1.0 else "random", noise_scale=noise_scale).requires_grad_(True)
optvars = [{'params': x}]
T_ref_lpyr = None
if optimizer_type == 'sgd':
    optimizer = torch.optim.SGD(optvars, lr=lr)
elif optimizer_type == 'adam':
    optimizer = torch.optim.Adam(optvars, lr=lr)
lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
# if warmup_iters > 0:
#     warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_iters)

# List to store frames for the video
frames = []
loss_history = []  # New: initialize loss history
mse_history = []
lap_mse_history = None 
for kk in range(num_iters):
    optimizer.zero_grad()
    loss0 = 0.0
    for b in range(batch_size_schedule[kk]):
        annealing_noise = simulated_annealing_schedule[kk]*torch.randn_like(x)
        pred = (x+annealing_noise).clamp(0., 1.)
        loss = loss_fn(pred, T_ref)#/batch_size_schedule[kk]
        loss0 += loss.item()/batch_size_schedule[kk]
        loss.backward()

    pred_lpyr, T_ref_lpyr = cvvdp.visualize_lpyr(x.clamp(0., 1.), T_ref, dim_order="CHW")

    # # Analytical gradient norm (L2 norm)
    # grad_norm = x.grad.norm().item()
    # # Finite difference gradient along the direction of x.grad:
    # epsilon = lr
    # # Compute the normalized direction of the gradient
    # grad_direction = x.grad / (x.grad.norm() + 1e-12)  # avoid division by zero
    # with torch.no_grad():
    #     # Perturb x along the gradient direction
    #     x_perturb = x.detach() + annealing_noise + epsilon * grad_direction
    #     # print(f'Perturbation Step Size: {grad_direction.norm(), (epsilon * grad_direction).norm(), (x + epsilon * grad_direction-x).norm()}')
    #     pred_perturb = x_perturb.clamp(0., 1.)
    #     loss_perturb = loss_fn(pred_perturb, T_ref)
    #     # Finite difference directional derivative approximation
    #     finite_grad = (loss_perturb - loss0) / epsilon
    #     finite_grad_norm = abs(finite_grad.item())
    # # Compute the error between the analytical gradient norm and the finite difference estimate
    # grad_norm_error = abs(grad_norm - finite_grad_norm)
    optimizer.step()

    # if warmup_iters > 0:
    #     with warmup_scheduler.dampening():
    #         if warmup_scheduler.last_step + 1 >= warmup_iters:
    #             lr_scheduler.step()

    if kk % WRITE_FREQ == 0:
        with torch.no_grad():
            loss_mse = mse_loss(x.clamp(0., 1.), T_ref).item()
            mse_history.append(loss_mse)
            if lap_mse_history is None:
                lap_mse_history = [[] for _ in range(len(pred_lpyr))]
            for n in range(len(pred_lpyr)):
                mse_val = torch.nn.functional.mse_loss(pred_lpyr[n], T_ref_lpyr[n]).item()
                lap_mse_history[n].append(mse_val)

            # Append current loss value
            loss_history.append(loss0)
            
            # ------------------ Added Visualization Code ------------------

            # Convert tensors to numpy arrays for visualization
            pred_np = to_imarray(x.clamp(0., 1.))
            # Convert overall error to a 2D array (averaging over channels)
            overall_err_np = to_imarray(x.clamp(0., 1.) - T_ref)

            global_vmin = np.min(overall_err_np)
            global_vmax = np.max(overall_err_np)
            
            # Determine the number of Laplacian levels and set the number of columns for the figure
            n_lap_levels = len(pred_lpyr)
            n_cols = max(3, n_lap_levels)  # top row: target, pred, overall error; bottom row: laplacian errors
            
            # Create a figure with two rows of subplots
            fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 8))
            
            # Top row: Target, Reconstruction, Overall Error (with colorbar)
            axes[0, 0].imshow(T_ref_np, cmap='gray' if T_ref_np.ndim == 2 else None)
            axes[0, 0].set_title('Target')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(pred_np, cmap='gray' if pred_np.ndim == 2 else None)
            axes[0, 1].set_title('Reconstruction')
            axes[0, 1].axis('off')
            
            im0 = axes[0, 2].imshow(np.mean(overall_err_np, axis=-1), cmap='viridis', vmin=global_vmin, vmax=global_vmax)
            axes[0, 2].set_title('Overall Error')
            axes[0, 2].axis('off')
            fig.colorbar(im0, ax=axes[0, 2])
            
            err_fft = np.log(np.mean(np.abs(np.fft.fftshift(np.fft.fft2(overall_err_np, axes=(0, 1)), axes=(0, 1))), axis=-1))
            im0 = axes[0, 3].imshow(err_fft, cmap='magma', vmin=np.min(err_fft), vmax=np.max(err_fft))
            axes[0, 3].set_title('Log Overall Error Spectrum')
            axes[0, 3].axis('off')
            fig.colorbar(im0, ax=axes[0, 3])
            
            ax_loss = fig.add_subplot(axes[0, 4])
            ax_loss.plot(loss_history, 'b-o')
            ax_loss.set_title("Loss", fontsize=10)
            ax_loss.set_xlabel("Iteration", fontsize=8)
            ax_loss.set_ylabel("Loss", fontsize=8)
            ax_loss.tick_params(labelsize=8)

            
            ax_loss = fig.add_subplot(axes[0, 5])
            ax_loss.plot(mse_history, 'b-o')
            ax_loss.set_title("MSE Loss", fontsize=10)
            ax_loss.set_xlabel("Iteration", fontsize=8)
            ax_loss.set_ylabel("Loss", fontsize=8)
            ax_loss.tick_params(labelsize=8)
            
            # Hide any extra subplots in the top row if n_cols > 6
            for j in range(6, n_cols):
                axes[0, j].axis('off')
            # Bottom row: Laplacian pyramid errors for each level (with the fixed colorbar range)
            for n in range(n_cols):
                ax = axes[1, n]
                if n < n_lap_levels:
                    # Remove extra batch dimension here as well
                    lap_err = np.mean(to_imarray(pred_lpyr[n][:,0,...] - T_ref_lpyr[n][:,0,...]), axis=-1)
                    im = ax.imshow(lap_err, cmap='viridis', vmin=global_vmin, vmax=global_vmax)
                    ax.set_title(f'Lap. Level {n}')
                    ax.axis('off')
                    fig.colorbar(im, ax=ax)
                else:
                    ax.axis('off')

            # ---- Third row: MSE loss per iteration for each Laplacian level ----
            for n in range(n_cols):
                ax = axes[2, n]
                if n < len(lap_mse_history):
                    ax.plot(lap_mse_history[n], 'r-o')
                    ax.set_title(f'Lap. Level {n} MSE')
                    ax.set_xlabel("Iteration", fontsize=8)
                    ax.set_ylabel("MSE", fontsize=8)
                    ax.tick_params(labelsize=8)
                else:
                    ax.axis('off')

            plt.suptitle(f'Iteration {kk+1} | Loss: {loss0:.4g} | MSE Loss: {loss_mse:.4g}', fontsize=16)

            if kk == 0:
                tic = time.time()
            toc = time.time()
            elapsed_time = toc - tic
            print(f'Iteration {kk+1} | Loss: {loss0:.4g} | MSE Loss: {loss_mse:.4g} | Elapsed Time: {elapsed_time:.2f} s')

            # Save the current figure as an image frame using the built-in io module
            buf = __import__('io').BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            frame = imageio.imread(buf)
            frames.append(frame)
            plt.close(fig)

            # ---------------- End of Visualization Code ----------------

# Save all frames as a video (here 10 fps, adjust as desired)
with imageio.get_writer(output_path, fps=10) as writer:
    for frame in frames:
        writer.append_data(frame)