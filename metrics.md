# Metrics

The code can run one of the following metrics (selected with `--metrics` or `-m` argument):

* `cvvdp`

  The latest version of ColorVideoVDP metric, with latest improvements.

  ColorVideoVDP v0.6.0 introduced a small change that has greatly improved metric performance when evaluating the results of tone mapping (see Sec. 3.3 in [this paper](http://dx.doi.org/10.1145/3799902.3811107)) and also improved stability when ColorVideoVDP is used for optimization. Prior to v0.6.0, ColorVideoVDP used the reference image to find the background luminance for contrast sensitivity. From v0.6.0 onwards, the sensitivity is computed separately for the test and reference images. Moreover, ColorVideoVDP 0.6.0 has been retrained on 4 datasets: [XR-DAVID](https://doi.org/10.17863/CAM.108210), [UPIQ](https://doi.org/10.17863/CAM.68372), [KADID10k](https://database.mmsp-kn.de/kadid-10k-database.html) and the dataset from QoMEX 2026 Challenge [Video Quality Assessment for Asymmetric Encoded Videos](https://sites.google.com/view/qomex26-vqm-gc/home).

* `cvvdp-0-5-6`

  The original ColorVideoVDP metric, as described in the [SIGGRAPH paper](https://doi.org/10.1145/3658144). The model parameters has not changed until version 0.5.6.

* `cvvdp-ml-saliency` [experimental]

  The extended version of ColorVideoVDP with a machine-learning based regressor and a saliency model, as explained in the [ICME paper](https://www.cl.cam.ac.uk/~rkm38/pdfs/hammou2025_ICME_GC_ColorVideoVDP_ML.pdf).

  This metric has been calibrated to perform well on the challenge dataset - video streaming distortions due to reduce bit-rate and resolution in both SDR and HDR content. The model was ranked the 3rd in the challenge, performing slightly worse than `cvvdp-ml-transformer`. 

  We do not recommend using this metric for optimization as it results a highly irregular loss landscape. It may also not generalize well to new (unseen) distortions. 

* `cvvdp-ml-transformer` [experimental]

  The extended version of ColorVideoVDP with a transformer, as explained in the [ICME paper](https://www.cl.cam.ac.uk/~rkm38/pdfs/hammou2025_ICME_GC_ColorVideoVDP_ML.pdf)

  This metric has been calibrated to perform well on the challenge dataset - video streaming distortions due to reduce bit-rate and resolution in both SDR and HDR content. The model was ranked the 2nd in the challenge.

  We do not recommend using this metric for optimization as it results a highly irregular loss landscape. It may also not generalize well to new (unseen) distortions. 

* `psnr-rgb`

  PSNR computed on the native RGB values. If HDR content is detected, the pixel values will be converted to the perceptually uniform spaces using [PU21](https://github.com/gfxdisp/pu21) encoding. Otherwise, the metric will operate like a regular PSNR. 

* `pu-psnr-rgb2020`

  PSNR computed on the RGB colour values, transformed to BT.2020 color spaces and represented using the perceptually uniform encoding [PU21](https://github.com/gfxdisp/pu21). It differs from psnr_rgb in two aspects: it will force transformation to BT.2020 and it will always apply PU21 encoding, even on SDR images.

* `pu-psnr-y`

  As above, but the PSNR is computed on the PU21-encoded luminance. 

* `dm-preview`, `dm-preview-exr`, `dm-preview-sbs`, `dm-preview-exr-sbs`

   A fake metric that outputs either HDR h.265 (.mp4) video (`dm-preview`) or OpenEXR frames (`dm-preview-exr`) with the output of the display model. It can be used to check or debug the display model. Use `--output-dir` to specify the directory in which the files should be written.




