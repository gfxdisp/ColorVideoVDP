# v0.4.2 (29/09/2024)
* Files are now sorted after the wildcard expansion
* Updated PU21 encoding parameters so that they are in sync with those in https://github.com/gfxdisp/pu21/
* Added: `--dump-channels` for generating videos with intermediate processing stages (debugging and visualization)
* Added: Support for HLG EOTF (e.g. iPhone HDR video) - thanks to Cosmin Stejerean
* Added: Processing of videos stored as image frames, described using the C-notation `frame_%04d.png`. New arguments: '--fps' and '--frames'
* Fixed: A better memory model for estimating how many frames can be processed at once on a GPU. Added '--gpu-mem' argument.
* Added: 'exposure' field in a display model JSON file.
* Added: `--result` argument to store results in a CSV file.
* Fixed: Added examples to README.md and improved documentation.
* Added: ColorVideoVDP logo.

# v0.4.1 (27/04/2024)
* Added `--metric dm-preview` for debugging of the display model
* Added a `loss` method to cvvdp and `examples/ex_adaptive_chroma_subsampling.py`
* Added "pixels_per_degree" as an optional field in `display_models.json`
* Platform is printed when running with `--verbose` (for reporting bugs/issues)

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