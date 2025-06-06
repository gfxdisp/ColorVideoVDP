# Example showing how to run ColorVideoVDP on a numpy array with a video
# The video is autogenerated in this example. See ex_aliasing.py for an example in which video is loaded from .mp4 files

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py

import os
import numpy as np
import time
import ex_utils as utils

import pycvvdp

'''
Results of current version (for reference):
Quality for blur-over-time: 8.829 JOD (took 8.9215 secs to compute)
'''

# The frame to use for the video
I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'tree.jpg'))

N = 60*4 # The number of frames
fps = 30 # Frames per second
sigma_max = 2

V_ref = np.repeat(I_ref[...,np.newaxis], N, axis=3) # Reference video (in color). 
SIGMAs = np.concatenate((np.linspace(0.01, sigma_max, N//2), np.linspace(sigma_max, 0.01, N//2)))
V_blur = utils.imgaussblur(V_ref, SIGMAs)

metric = pycvvdp.cvvdp(display_name='standard_4k', heatmap=None)

start = time.time()
Q_JOD, stats = metric.predict(V_blur, V_ref, dim_order="HWCF", frames_per_second=fps)
end = time.time()

print(f'Quality for blur-over-time: {Q_JOD:.3f} JOD (took {end-start:.4f} secs to compute)')
