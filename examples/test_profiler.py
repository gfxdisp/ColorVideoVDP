# Profile PyTorch execution
import torch
import os
import glob
import time

from torch.profiler import profile, record_function, ProfilerActivity

import pycvvdp
from pycvvdp.video_source import video_source_array
from pycvvdp.display_model import vvdp_display_photo_eotf

display_name = 'standard_hdr_linear'
media_folder = os.path.join(os.path.dirname(__file__), '..',
                            'example_media', 'aliasing')

# Add here paths to the test and reference videos
#tst_fname = 'S:\\Datasets\\LIVEHDR\\test\\4k_6M_Bonfire.mp4'
# tst_fname = 'S:\\Datasets\\LIVEHDR\\test\\720p_4.6M_Bonfire.mp4'
# ref_fname = 'S:\\Datasets\\LIVEHDR\\test\\4k_ref_Bonfire.mp4'
tst_fname = 'example_media/aliasing/ferris-bicubic-bicubic.mp4'
ref_fname = 'example_media/aliasing/ferris-ref.mp4'
# tst_fname = 'S:\\Datasets\\color_display_quality\\Bonfire_Blur_Level003.mp4'
# ref_fname = 'S:\\Datasets\\color_display_quality\\Bonfire_reference_Level001.mp4'

# tst_fname = 'S:\\Datasets\\color_display_quality\\Business_ColorFringes_Level003.mp4'
# ref_fname = 'S:\\Datasets\\color_display_quality\\Business_reference_Level001.mp4'


metric = pycvvdp.cvvdp(display_name=display_name, heatmap=None)

frames = 30

# preload=True will load all the frames into memory and allow random access
vs_file = pycvvdp.video_source_file( tst_fname, ref_fname, display_photometry=display_name, frames=frames, preload=True)

print( f"Pre-loading {frames} frames..." )
start = time.time()
# Getting a single frame should trigger pre-load
vs_file.get_test_frame( 0, metric.device )
end = time.time()
print( 'Loading frames took {:.4f} secs'.format(end-start) )

H, W, N = vs_file.get_video_size()
tst_frames = torch.zeros( [1, 1, frames, H, W], dtype=torch.float32, device=metric.device )
ref_frames = torch.zeros( [1, 1, frames, H, W], dtype=torch.float32, device=metric.device )

print( "Transferring frames to the GPU..." )
start = time.time()
for ff in range(frames):
    tst_frames[:,:,ff,:,:] = vs_file.get_test_frame( ff, metric.device )
    ref_frames[:,:,ff,:,:] = vs_file.get_reference_frame( ff, metric.device )
end = time.time()
print( 'Transferring frames took {:.4f} secs'.format(end-start) )

# Using fvvdp_display_photo_absolute as display model has been already applied
vs = video_source_array( tst_frames, ref_frames, vs_file.get_frames_per_second(), display_photometry=vvdp_display_photo_eotf( Y_peak=10000, EOTF='linear') )

del vs_file # Explicitly close video reading processes

print( "Running the metric..." )
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        start = time.time()
        Q_JOD_static, stats_static = metric.predict_video_source( vs )
        end = time.time()

#print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=20))
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
#print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
prof.export_chrome_trace("trace.json")

print( 'Quality for {}: {:.3f} JOD (took {:.4f} secs to compute)'.format(tst_fname, Q_JOD_static, end-start) )
