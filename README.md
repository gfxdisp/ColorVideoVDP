# ColourVideoVDP: A visible difference predictor for colour images and videos

**[TODO:]** Teaser

ColourVideoVDP is a full-reference visual quality metric that predicts the perceptual difference between pairs of images or videos. Similar to popular metrics like PSNR and SSIM, it is aimed at comparing a ground truth reference video against a distorted (e.g. compressed, lower framerate) version. However, unlike traditional quality metrics, ColourVideoVDP is based on fundamental perceptual models of contrast sensitivity and masking. 

The main features:
* models chromatic and achromatic contrast sensitivity;
* models spatio-temporal sensitivity so it can predict visibility of flicker and other temporal artifacts;
* works with colorimetrically calibrated content, both SDR and HDR (any colour space);
* can predict a single number quality correlate or a distortion map.

ColourVideoVDP is implemented in PyTorch and can be run efficiently on a CUDA-enabled GPU. It can also run on a CPU, but the processing times will be much larger, especially for video. Its usage is described [below](#usage).

The details of the metric can be found in:

> ColorVideoVDP: A Visible Difference Predictor for Images and Video.
> Rafal K. Mantiuk, Param Hanji, Maliha Ashraf, Alexandre Chapiro, Yuta Asano.
> Paper in preparation.

**[TODO:]** Link to project page

If you use the metric in your research, please cite the paper above. 

## PyTorch quickstart
Start by installing [anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Then, create a new environment for ColourVideoVDP and activate it:
```bash
conda create -n cvvdp python=3.10
conda activate cvvdp
```

Install PyTorch v1.X by following [these instructions](https://pytorch.org/get-started/previous-versions/#v1131). **If you have an Nvidia GPU with appropriate drivers, it is recommended to install with conda for proper CUDA support**. Finally, clone this repository (or extract from a .zip file), and install ColourVideoVDP with PyPI:

```bash
git clone git@github.com:mantiuk/ColourVideoVDP.git   # skip if a .zip is provided
cd ColourVideoVDP
pip install -e .
```
*Note:* The "-e/--editable" option to `pip` is optional and should be used only if you intend to change the ColourVideoVDP code.

After installation, run `cvvdp` directly from the command line:

```bash
cvvdp --test test_file --ref ref_file --display standard_fhd
```
The test and reference files can be images or videos. The option `--display` specifies a display on which the content is viewed. See [vvdp_data/display_models.json](https://github.com/mantiuk/ColourVideoVDP/blob/main/pycvvdp/vvdp_data/display_models.json) for the available displays.

Note that the default installation skips the [PyEXR](https://pypi.org/project/PyEXR/) package and uses ImageIO instead. It is recommended to separately install this package since ImageIO's handling of OpenEXR files is unreliable as evidenced [here](https://github.com/imageio/imageio/issues/517). PyEXR is not automatically installed because it depends on the [OpenEXR](https://www.openexr.com/) library, whose installation is operating system specific.

See [Command line interface](#command-line-interface) for further details. ColourVideoVDP can be also run directly from Python - see [Low-level Python interface](#low-level-python-interface). 

**Table of contents**
- [Display specification](#display-specification)
    - [Custom specification](#custom-specification)
    - [HDR content](#HDR-content)
    - [Reporting metric results](#reporting-metric-results)
    - [Predicting quality scores](#predicted-quality-scores)
- [Usage](#usage)
    - [Command line interface](#command-line-interface)
    - [Low-level Python interface](#low-level-python-interface)
- [Release notes](#release-notes)

## Display specification

Unlike most image quality metrics, ColourVideoVDP needs physical specification of the display (e.g. its size, resolution, peak brightness) and viewing conditions (viewing distance, ambient light) to compute accurate predictions. The specifications of the displays are stored in [vvdp_data/display_models.json](https://github.com/mantiuk/ColourVideoVDP/blob/main/pycvvdp/vvdp_data/display_models.json). You can add the exact specification of your display to this file, or create a new JSON file and pass the directory it is located in as `--config-dir` parameter (`cvvdp` command). If the display specification is unknown to you, you are encouraged to use one of the standard display specifications listed on the top of that file, for example `standard_4k`, or `standard_fhd`. If you use one of the standard displays, there is a better chance that your results will be comparable with other studies. 

You specify the display by passing `--display` argument to `cvvdp`.

Note the the specification in `display_models.json` is for the display and not the image. If you select to use `standard_4k` with the resolution of 3840x2160 for your display and pass a 1920x1080 image, the metric will assume that the image occupies one quarter of that display (the central portion). If you want to enlarge the image to the full resolution of the display, pass `--full-screen-resize {fast_bilinear,bilinear,bicubic,lanczos}` option (for now it works with video only). 

The command line version of ColourVideoVDP can take as input HDR video streams encoded using the PQ transfer function. To correctly model HDR content, it is necessary to pass a display model with `EOTF="PQ"`, for example `standard_hdr_pq`.

### Custom display specification

If you run the metric from the command line, we recommend that you create a directory with a copy of `display_models.json`, add a new display specification in that file and then add to the command line `--config-dir <path-to-dir-with-json-file> --display <name-of-display-spec>`.

If you run the metric from Python code, the display photometry and geometry can be specified by passing `display_name` parameter to the metric. Alternatively, if you need more flexibility in specifying display geometry (size, viewing distance) and its colorimetry, you can instead pass objects of the classes `vvdp_display_geometry`, `vvdp_display_photo_gog` for most SDR displays, and `vvdp_display_photo_absolute` for HDR displays. You can also create your own subclasses of those classes for custom display specification. 

### HDR content

(Python command line only) You can use the metric to compare: 

* HDR video files encoded using PQ EOTF function (SMPTE ST 2084). Pass the video files as `--test` and `--ref` arguments and specify `--display standard_hdr_pq`.

* OpenEXR images. The images *MUST* contain absolute linear colour values (colour graded values, emitted from the display). That is, if the disply peak luminance is 1000, RGB=(1000,1000,1000) corresponds to the maximum value emitted from the display. If you pass images with the maximum value of 1, the metric will assume that the images are very dark (the peak of 1 nit) and result in incorerect predictrions. You need to specify `--display standard_hdr_linear` to use correct EOTF.

### Reporting metric results

When reporting the results of the metric, please include the string returned by the metric, such as:
`"ColourVideoVDP v0.1, 75.4 [pix/deg], Lpeak=200, Lblack=0.5979 [cd/m^2], (standard_4k)"`
This is to ensure that you provide enough details to reproduce your results. 

### Predicted quality scores

ColourVideoVDP reports image/video quality in the JOD (Just-Objectionable-Difference) units. The highest quality (no difference) is reported as 10 and lower values are reported for distorted content. In case of very strong distortion, or when comparing two unrelated images, the quality value can drop below 0. 

The main advantage of JODs is that they (a) should be linearly related to the perceived magnitude of the distortion and (b) the difference of JODs can be interpreted as the preference prediction across the population. For example, if method A produces a video with the quality score of 8 JOD and method B gives the quality score of 9 JOD, it means that 75% of the population will choose method B over A. The plots below show the mapping from the difference between two conditions in JOD units to the probability of selecting the condition with the higher JOD score (black numbers on the left) and the percentage increase in preference (blue numbers on the right).

## Usage

### Command line interface
The main script to run the model on a set of images or videos is [run_cvvdp.py](https://github.com/gfxdisp/ColourVideoVDP/blob/main/pycvvdp/run_cvvdp.py), from which the binary `cvvdp` is created . Run `cvvdp --help` for detailed usage information.

For the first example, a video was downsampled (4x4) and upsampled (4x4) by different combinations of Bicubic and Nearest filters. To predict quality, you can run:

```bash
cvvdp --test example_media/aliasing/ferris-*-*.mp4 --ref example_media/aliasing/ferris-ref.mp4 --display standard_fhd --heatmap supra-threshold
```

|Original | ![ferris wheel](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/aliasing/ferris-ref.gif) | Quality | **TODO:** Difference map |
| :---: | :---: | :---: | :---: |
| Bicubic &#8595;<br />Bicubic &#8593;<br />(4x4) | ![bicubic-bicubic](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/aliasing/ferris-bicubic-bicubic.gif) | 7.0957 | |
| Bicubic &#8595;<br />Nearest &#8593;<br />(4x4) | ![bicubic-nearest](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/aliasing/ferris-bicubic-nearest.gif) | 6.9652 | |
| Nearest &#8595;<br />Bicubic &#8593;<br />(4x4) | ![nearest-bicubic](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/aliasing/ferris-nearest-bicubic.gif) | 7.0256 | |
| Nearest &#8595;<br />Nearest &#8593;<br />(4x4) | ![nearest-nearest](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/aliasing/ferris-nearest-nearest.gif) | 6.9634 | |


### Low-level Python interface
ColourVideoVDP can also be run through the Python interface by instatiating the `pycvvdp.cvvdp` class. This example shows how to predict the quality of images degraded by Gaussian noise and blur.

```python
import pycvvdp
import numpy as np
import os.path as osp
import pytorch_examples.ex_utils as utils

I_ref = pycvvdp.load_image_as_array(osp.join('example_media', 'wavy_facade.png'))
metric = pycvvdp.cvvdp(display_name='standard_4k', heatmap='threshold')

# Gaussian noise with variance 0.003
I_test_noise = utils.imnoise(I_ref, np.sqrt(0.003))
Q_JOD_noise, stats_noise = metric.predict( I_test_noise, I_ref, dim_order="HWC" )

# Gaussian blur with sigma=2
I_test_blur = utils.imgaussblur(I_ref, 2)
Q_JOD_blur, stats_blur = metric.predict( I_test_blur, I_ref, dim_order="HWC" )
```

|Original | ![wavy-facade](https://github.com/gfxdisp/FovVideoVDP/raw/main/example_media/wavy_facade.png) | Quality | **TODO: **Difference map |
| :---: | :---: | :---: | :---: |
| Gaussian noise (σ<sup>2</sup> = 0.003) | ![noise](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/simple_image/wavy_facade_noise.png) | 9.0533 | |
| Gaussian blur (σ = 2) | ![blur](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/simple_image/wavy_facade_blur.png) | 7.9508 | |

More examples can be found in these [example scripts](https://github.com/gfxdisp/ColourVideoVDP/blob/main/pytorch_examples).


## Release notes

* v0.1.0 - ?

The detailed list of changes can be found in [ChangeLog.md](https://github.com/gfxdisp/ColourVideoVDP/blob/main/ChangeLog.md).
