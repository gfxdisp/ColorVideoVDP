# Saving detailed results

To get more detailed breakdown of ColorVideoVDP scores, you can pass `--results-detailed` argument to `cvvdp` or call `cvvdp.write_results_detailed(stats, fname)` method. In either case, ColorVideoVDP will write a NumPy `<name>.npy` file containing a dictionary with the following items:

* `cfb_map` - a NumPy array of the size [channels, frames, bands] with the visual difference scores. 
Note that these are differences so that a larger value means a larger visual difference. Those are *not* scaled in the JOD units. Channels are written in the order: achromatic sustained, chromatic red-green, chromatic yellow-violet, achromatic transient. The last achromatic transient channel is skipped for images. The bands correspond to different spatual frequencies, which are listed in the field `rho_band`. 
* `rho_band` - a list of the size bands, with the peak spatial frequency of each band in cycles per degree.
* `frames_per_second` - frame rate (0 for images)
* `width` - width of the original video/image frame
* `height` - height of the original video/image frame
* `N_frames` - the number of frames

Note that you need to allow pickers when loadining the detailed results file:
```
res = np.load( "per_frame_res_cvvdp.npy", allow_pickle=True )
``` 

## The interpretation of cfb_map values

The values in the `cfb_map` correspond to per-chanel, per-frame and per-band visual differences, pooled across all the pixels, weighted by the learned weights and converted to JOD units. These are the same values as those used to create distograms. No difference is mapped to 0 and the values increase with larger visual differences. 

Please note that the reported JOD units are only approximate because there is no data to with per-frame and per-channel JODs that could validate those predictions. Averaging those values will *not* give the final JOD score as ColorVideoVDP uses more complicated pooling strategy, with different p-norms used for different dimensions (see eq. (14) in the ColorVideoVDP paper). 

## File names

`--results-detailed` argument must be followed by the base file name, with or without a path. Do not add `npy` extension as it will be added automatically. If the command runs on more than one content, the test content name will be added to the results file name. If more than one metric is run, the metric name will be added to the result file name. 
