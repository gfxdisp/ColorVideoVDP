import torchvision.transforms
import torch
import torch.nn.functional as F
import pycvvdp.utils as utils
import pycvvdp
import os
import numpy as np
import matplotlib.pyplot as plt

device = 'cpu'
I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
img_tensor = torch.as_tensor(I_ref.astype(np.float32)).to(device).unsqueeze(0).permute(3, 0, 1, 2) / (2**16 - 1)

img_tensor = F.interpolate(img_tensor, size=(100, 100), mode='bilinear', align_corners=False)

## Same filter size to handle uncertainty
sigma = 3
kernel_size = int(sigma*4)+1
half_kernel = kernel_size // 2

Gf_reflect_pad = utils.ImGaussFilt(sigma, device=device, mode='reflect')
filtered_custom_reflect = Gf_reflect_pad.run_4d(img_tensor)

Gf_replicate_pad = utils.ImGaussFilt(sigma, device=device, mode='replicate')
filtered_custom_replicate = Gf_replicate_pad.run_4d(img_tensor)

torch_gb = torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
filtered_torch = torch_gb.forward(img_tensor)

padding = (half_kernel, half_kernel, half_kernel, half_kernel)  
filtered_custom_reflect_padded = F.pad(filtered_custom_reflect, padding, mode='constant', value=0)
filtered_custom_replicate_padded = F.pad(filtered_custom_replicate, padding, mode='constant', value=0)
filtered_torch_padded = F.pad(filtered_torch, padding, mode='constant', value=0)

difference_reflect = torch.abs(filtered_custom_reflect_padded - filtered_torch_padded).squeeze().permute(1, 2, 0).cpu().numpy().astype(np.float32)
difference_replicate = torch.abs(filtered_custom_replicate_padded - filtered_torch_padded).squeeze().permute(1, 2, 0).cpu().numpy().astype(np.float32)


fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(filtered_custom_reflect_padded.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.float32), cmap='hot', vmin=0, vmax=1)
axes[0, 0].set_title('Filtered Custom Reflect')
axes[0, 0].axis('off')
axes[0, 1].imshow(difference_reflect, cmap='hot', vmin=0, vmax=1e-9)
axes[0, 1].set_title(f'Difference Custom Reflect vs Torch (Sum: {np.sum(difference_reflect):.2e})')
axes[0, 1].axis('off')
axes[1, 0].imshow(filtered_custom_replicate_padded.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.float32), cmap='hot', vmin=0, vmax=1)
axes[1, 0].set_title('Filtered Custom Replicate')
axes[1, 0].axis('off')
axes[1, 1].imshow(difference_replicate, cmap='hot', vmin=0, vmax=1e-9)
axes[1, 1].set_title(f'Difference Custom Replicate vs Torch (Sum: {np.sum(difference_replicate):.2e})')
axes[1, 1].axis('off')

# plt.colorbar()
plt.tight_layout()
plt.show()
