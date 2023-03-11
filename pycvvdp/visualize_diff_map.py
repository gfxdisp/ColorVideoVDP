# Visualization of difference maps.

import torch
from pycvvdp.interp import interp1

# For debugging only
# from gfxdisp.pfs.pfs_torch import *

def luminance_NCHW(x):
    if x.shape[1] == 3: # NC***
        y = (
            x[:,0:1,...] * 0.212656 + 
            x[:,1:2,...] * 0.715158 + 
            x[:,2:3,...] * 0.072186)
    else:
        y = x

    return y

def log_luminance(x):
    y = luminance_NCHW(x)
    clampval = torch.min(y[y>0.0])
    return torch.log(torch.clamp(y, min=clampval))


def vis_tonemap(b, dr):
    t = 3.0
    
    b_min = torch.min(b)
    b_max = torch.max(b)
    
    if b_max-b_min < dr: # No tone-mapping needed
        tmo_img = (b/(b_max-b_min+1e03)-b_min)*dr + (1-dr)/2
        return tmo_img

    b_scale = torch.linspace( b_min, b_max, 1024, device=b.device)
    b_p = torch.histc( b, 1024, b_min, b_max )
    b_p = b_p / torch.sum(b_p)
    
    sum_b_p = torch.sum(torch.pow(b_p, 1.0/t))

    dy = torch.pow(b_p, 1.0/t) / sum_b_p
    
    v = torch.cumsum(dy, 0)*dr + (1.0-dr)/2.0
    
    tmo_img = interp1(b_scale, v, b)

    return tmo_img

# returns sRGB image/frames
def visualize_diff_map(diff_map, context_image=None, colormap_type="supra-threshold"):
    diff_map = torch.clamp(diff_map, 0.0, 1.0)

    if context_image is None:
        tmo_img = torch.ones_like(diff_map) * 0.5
    else:
        tmo_img = vis_tonemap(log_luminance(context_image), 0.6)

    if colormap_type == 'threshold':
        # Visualize up to 1 JOD (>=1 JOD will be all red)

        color_map = torch.tensor([
            [0.2, 0.2, 1.0],
            [0.2, 1.0, 1.0],
            [0.2, 1.0, 0.2],
            [1.0, 1.0, 0.2],
            [1.0, 0.2, 0.2],
        ], device=diff_map.device)
        color_map_in = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00], device=diff_map.device)*0.1

    elif colormap_type == 'supra-threshold':
        # Visualize up to 3 JOD (>=3 JOD will be all yellow)

        color_map = torch.tensor([
            [0.2, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.2],
        ], device=diff_map.device)
        color_map_in = torch.tensor([0.0, 0.5, 1.0], device=diff_map.device)*0.3

    elif colormap_type == 'monochromatic':
        
        color_map = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], device=diff_map.device)
        
        color_map_in = torch.tensor([0.0, 1.0], device=diff_map.device)
    else:
        print("Unknown colormap: %s" % colormap_type)

    # cmap = torch.zeros_like(diff_map)
    # if cmap.shape[1] == 1:
    #     cmap = torch.cat([cmap]*3, 1)
    frame_count, h, w = diff_map.shape[-3], diff_map.shape[-2], diff_map.shape[-1]
    cmap = torch.empty( [3, frame_count, h, w], device=diff_map.device, dtype=torch.float16)

    color_map_l = color_map[:,0:1] * 0.212656 + color_map[:,1:2] * 0.715158 + color_map[:,2:3] * 0.072186
    color_map_ch = color_map / (torch.cat([color_map_l] * 3, 1) + 0.0001)

    cmap[0:1,...] = interp1(color_map_in, color_map_ch[:,0], diff_map).type(torch.float16)
    cmap[1:2,...] = interp1(color_map_in, color_map_ch[:,1], diff_map).type(torch.float16)
    cmap[2:3,...] = interp1(color_map_in, color_map_ch[:,2], diff_map).type(torch.float16)

    cmap = (cmap * tmo_img).clip(0.,1.)

    return cmap