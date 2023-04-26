# v0.1.1 
* Heatmaps are now saved using ffmpeg's mpeg4 codec for better comparibility across platforms
* Fixed freezing on Windows when reading long videos (due to bug in python's /dev/null implementation)
* Added plain psnr_rgb metric, which operates on display-encoded values (or PU21-encoded if needed)
* Updated PU-PSNR-* to use 100 nit as the peak + to computed PSNR for all pixels in the video (rather than mean PSNR over all frames).

# v0.1.0 Initial beta release (27 March 2023)