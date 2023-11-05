# v0.? (?)
* Added Matlab wrapper

# v0.3 (05/09/2023)
* Added value range checks when the metric is running on HDR data (to avoid passing relative values)
* Added SSIM as an alternative metric
* Better handling of paths to configuration files

# v0.2
* Updated ColourVideoVDP model with cross-channel masking and more advanced pooling, different calibration and better prediction accuracy.
* Added distograms
* Changed handling of paths to configuration files

# v0.1.1 
* Heatmaps are now saved using ffmpeg's mpeg4 codec for better comparibility across platforms
* Fixed freezing on Windows when reading long videos (due to bug in python's /dev/null implementation)
* Added plain psnr_rgb metric, which operates on display-encoded values (or PU21-encoded if needed)
* Updated PU-PSNR-* to use 100 nit as the peak + to computed PSNR for all pixels in the video (rather than mean PSNR over all frames).
* Reorganized display_models.json - now color space is a part of the display model json spec, EOTF is a part of color space json spec.

# v0.1.0 Initial beta release (27 March 2023)