# v0.5 (?)
* Added `--metric dm-preview` for debugging of the display model

# v0.4 (19/01/2024)
* A new calibration with a small improvement in performance
* Improved predictions for supra-threshold contrast across color directions
* Improved masking model
* Added Matlab wrapper
* CLI now has --interactive mode to process multiple images/video without restarting PyTorch
* Fixed distogram generation for images
* "--quiet" flag now ensures that no warning messages end up in the stdout
* CSF updated to the latest castleCSF fit
* added 'luminance' color space to handle luminance-only data

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