import numpy as np
import pandas as pd
import pycvvdp
from time import time
import torch
from tqdm import tqdm, trange

class DummyVS:
    def __init__(self, num_frames, height, width, fps=30, device='cuda'):
        self.n = num_frames
        self.h = height
        self.w = width
        self.fps = fps
        self.frame = torch.randn(1,3,1,height,width, device=device)

    def __len__(self):
        return self.n
    
    def get_video_size(self):
        return self.h, self.w, self.n

    def get_frames_per_second(self):
        return self.fps

    def get_reference_frame(self, i, device=None, colorspace=None):
        return self.frame[:,:1] if colorspace == 'Y' else self.frame

    def get_test_frame(self, i, device=None, colorspace=None):
        return self.get_reference_frame(i, device, colorspace)

    def to(self, device):
        self.frame = self.frame.to(device)

metrics = ['cvvdp', 'cvvdp-cpu', 'fvvdp', 'PSNR-RGB']
device = torch.device('cuda')
dims = ((1080, 1920), (720, 1280), (1440, 2560), (2160, 3840), (4320, 7680))
# h, w = 1080, 1920
# n = (30, 60, 120, 240, 480)
num_frames = 1
num_samples = 5

# VMAF paths set for Param's PC
ffmpeg_path = '../vmaf/ffmpeg-6.0-amd64-static/ffmpeg'
vmaf_cache = '/local/scratch/pmh64/tmp'

timings = pd.DataFrame(columns=['metric', 'num_frames', 'height', 'width', 'time'])
timings = []
# for num_frames in tqdm(n):
for h, w in tqdm(dims):
    vs = DummyVS(num_frames, h, w, device=device)
    pbar = tqdm(metrics, leave=False)
    for metric in pbar:
        pbar.set_description(f'N={num_frames}, metric={metric}')
        if metric.endswith('cpu'):
            vs.to('cpu')
        else:
            vs.to(device)

        if metric == 'cvvdp':
            metric = pycvvdp.cvvdp(quiet=True, device=device, temp_padding='replicate', heatmap=None)
        elif metric == 'cvvdp-cpu':
            metric = pycvvdp.cvvdp(quiet=True, device=torch.device('cpu'), temp_padding='replicate', heatmap=None)
        elif metric == 'fvvdp':
            from pyfvvdp import fvvdp
            # Add argument "colorspace='Y'" while retriving frames to run on images
            # Lines 233 and 244 in commit 5bf67f92341604d238ebe72fdeeb4ad825db5485
            metric = fvvdp(quiet=True, device=device, temp_padding='replicate', heatmap=None)
        elif metric == 'PSNR-RGB':
            metric = pycvvdp.psnr_rgb(device=device)
        elif metric == 'FLIP':
            metric = pycvvdp.flip(device=device)
        elif metric == 'PU21-VMAF':
            metric = pycvvdp.pu_vmaf(ffmpeg_path, vmaf_cache, device=device)
        else:
            raise RuntimeError( f'Unknown metric {metric}' )

        with torch.no_grad():
            metric.predict_video_source(vs)     # dummy run

            times = []
            for _ in trange(num_samples, leave=False):
                start = time()
                metric.predict_video_source(vs)
                times.append(time() - start)

        timings.append({'name': metric.short_name(),
                        'num_samples': num_samples,
                        'num_frames': num_frames,
                        'height': h,
                        'width': w,
                        'time_mean': np.mean(times),
                        'time_std': np.std(times)})

df = pd.DataFrame(timings)
df.to_csv('timings.csv', index=False)
