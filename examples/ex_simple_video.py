# Example showing how to run ColorVideoVDP on numpy arrays with videos. 

# The video is generated in the code in this example. See ex_aliasing.py for an example in which video is loaded from .mp4 files

# Important: This and other examples should be executed from the main ColorVideoVDP directory:
# python examples/ex_<...>.py


import os
import time
import numpy as np
import ex_utils as utils

import pycvvdp

'''
Results of current version (for reference):
Quality for static noise: 8.793 JOD (took 2.4667 secs to compute)
Quality for dynamic noise: 9.307 JOD (took 1.4988 secs to compute)
'''

# The frame to use for the video
I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))

N = 60 # The number of frames
fps = 30 # Frames per second

V_ref = np.repeat(I_ref[...,np.newaxis], N, axis=3) # Reference video (in colour). 
N_amplitude = 0.07; # Amplitude of the noise (in gamma encoded values, scale 0-1)
V_static_noise = utils.imnoise(V_ref, N_amplitude, static=True)
V_dynamic_noise = utils.imnoise(V_ref, N_amplitude)

metric = pycvvdp.cvvdp(display_name='standard_4k', heatmap=None)

start = time.time()
Q_JOD_static, stats_static = metric.predict( V_static_noise, V_ref, dim_order="HWCF", frames_per_second=fps )
end = time.time()

print( 'Quality for static noise: {:.3f} JOD (took {:.4f} secs to compute)'.format(Q_JOD_static, end-start) )

start = time.time()
Q_JOD_dynamic, stats_dynamic = metric.predict( V_dynamic_noise, V_ref, dim_order="HWCF", frames_per_second=fps )
end = time.time()

print( 'Quality for dynamic noise: {:.3f} JOD (took {:.4f} secs to compute)'.format(Q_JOD_dynamic, end-start) )
