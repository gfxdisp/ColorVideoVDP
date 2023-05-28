# ColourVideoVDP: A visible difference predictor for colour images and videos

**[BETA RELEASE]**
> **[in the Meta version we have some beta disclaimers]**

ColourVideoVDP is a full-reference visual quality metric that predicts the perceptual difference between pairs of images or videos. Similar to popular metrics like PSNR, SSIM, and DeltaE 2000 it is aimed at comparing a ground truth reference against a distorted (e.g. blurry, noisy, color-shifted) version. 

This metric is unique because it is the first color-aware metric that accounts for spatial and temporal aspects of vision. 

The main features:
* models chromatic and achromatic contrast sensitivity using a novel contrast sensitivity model, allowing us to predict distortions in colour and luminance;
* models spatio-temporal sensitivity so that it can predict visibility of artifacts like waveguide nonuniformity and other temporally varying artifacts;
* works with colorimetrically calibrated content, both SDR and HDR (any colour space);
* can predict a single number quality correlate or a distortion map.

ColourVideoVDP is implemented in PyTorch and can be run efficiently on a CUDA-enabled GPU. It can also run on a CPU, but the processing times will be much longer, especially for video. Its usage is described [below](#example-usage).

The details of the metric will be available in our upcoming paper:

> ColorVideoVDP: A Visible Difference Predictor for Images and Video.
> Rafal K. Mantiuk, Param Hanji, Maliha Ashraf, Yuta Asano, Alexandre Chapiro.
> Paper in preparation.

**[TODO:]** Link to project page

If you use the metric in your research, please cite the paper above. 

## PyTorch quickstart
1. Start by installing [anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Then, create a new environment for ColourVideoVDP and activate it:
```bash
conda create -n cvvdp python=3.10
conda activate cvvdp
```

2. Install PyTorch by following [these instructions](https://pytorch.org/get-started/locally/) (OS-specific). **If you have an Nvidia GPU with appropriate drivers, it is recommended to install with conda for proper CUDA support**. To use MPS on a Mac, please install torch>=2.1.0.

3. Install [ffmpeg](https://ffmpeg.org/). The easiest option is to install using conda,
```bash
conda install ffmpeg
```

4. Obtain the ColourVDP codebase, by extracting a `.zip` file provided or cloning from Github:
```bash
git clone git@github.com:mantiuk/ColourVideoVDP.git   # skip if a .zip is provided or you use Github GUI
```

5. Finally, install ColourVideoVDP with PyPI:
```bash
cd ColourVideoVDP
pip install -e .
```
*Note:* The "-e/--editable" option to `pip` is optional and should be used only if you intend to change the ColourVideoVDP code.

After installation, run `cvvdp` directly from the command line:

```bash
cvvdp --test test_file --ref ref_file --display standard_fhd
```
The test and reference files can be images or videos. The option `--display` specifies a display on which the content is viewed. See [vvdp_data/display_models.json](pycvvdp/vvdp_data/display_models.json) for the available displays.

See [Command line interface](#command-line-interface) for further details. ColourVideoVDP can be also run directly from Python - see [Low-level Python interface](#low-level-python-interface). 

**Table of contents**
- [Display specification](#display-specification)
    - [Custom specification](#custom-display-specification)
    - [HDR content](#hdr-content)
    - [Reporting metric results](#reporting-metric-results)
    - [Predicting quality scores](#predicted-quality-scores)
- [Usage](#example-usage)
    - [Command line interface](#command-line-interface)
    - [Low-level Python interface](#low-level-python-interface)
- [Release notes](#release-notes)

## Display specification

Unlike most image quality metrics, ColourVideoVDP needs physical specification of the display (e.g. its size, resolution, peak brightness) and viewing conditions (viewing distance, ambient light) to compute accurate predictions. The specifications of the displays are stored in [vvdp_data/display_models.json](pycvvdp/vvdp_data/display_models.json). You can add the exact specification of your display to this file, or create a new JSON file and pass the directory it is located in as `--config-dir` parameter (`cvvdp` command). If the display specification is unknown to you, you are encouraged to use one of the standard display specifications listed on the top of that file, for example `standard_4k`, or `standard_fhd`. If you use one of the standard displays, there is a better chance that your results will be comparable with other studies. 

You specify the display by passing `--display` argument to `cvvdp`.

Note the the specification in `display_models.json` is for the display and not the image. If you select to use `standard_4k` with the resolution of 3840x2160 for your display and pass a 1920x1080 image, the metric will assume that the image occupies one quarter of that display (the central portion). If you want to enlarge the image to the full resolution of the display, pass `--full-screen-resize {fast_bilinear,bilinear,bicubic,lanczos}` option (for now it works with video only). 

The command line version of ColourVideoVDP can take as input HDR video streams encoded using the PQ transfer function. To correctly model HDR content, it is necessary to pass a display model with correct color space and transfer function (with the field `colorspace="BT.2020-PQ"`), for example `standard_hdr_pq`.

The display specification is documented in [vvdp_data/README.md](pycvvdp/vvdp_data/README.md).

### Custom display specification

If you run the metric from the command line, we recommend that you create a directory with a copy of `display_models.json`, add a new display specification in that file and then add to the command line `--config-dir <path-to-dir-with-json-file> --display <name-of-display-spec>`.

If you run the metric from Python code, the display photometry and geometry can be specified by passing `display_name` parameter to the metric. Alternatively, if you need more flexibility in specifying display geometry (size, viewing distance) and its colorimetry, you can instead pass objects of the classes `vvdp_display_geometry`, `vvdp_display_photo_gog` for most SDR displays, and `vvdp_display_photo_absolute` for HDR displays. You can also create your own subclasses of those classes for custom display specification. 

### HDR content

(Python command line only) You can use the metric to compare: 

* HDR video files encoded using PQ EOTF function (SMPTE ST 2084). Pass the video files as `--test` and `--ref` arguments and specify `--display standard_hdr_pq`.

* OpenEXR images. The images *MUST* contain absolute linear colour values (colour graded values, emitted from the display). That is, if the disply peak luminance is 1000, RGB=(1000,1000,1000) corresponds to the maximum value emitted from the display. If you pass images with the maximum value of 1, the metric will assume that the images are very dark (the peak of 1 nit) and result in incorerect predictrions. You need to specify `--display standard_hdr_linear` to use correct EOTF. Note that the default installation skips the [PyEXR](https://pypi.org/project/PyEXR/) package, which is required to read `.exr` files. To install, run:
```bash
conda install -c conda-forge openexr-python.   # or "sudo apt install openxr" on Linux machines
pip install pyexr
```
**Troubleshooting on Linux:** You may need to update your library path by adding the following line to your `~/.bashrc`:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/conda/miniconda3/lib
```

### Reporting metric results

When reporting the results of the metric, please include the string returned by the metric, such as:
`"ColourVideoVDP v0.1, 75.4 [pix/deg], Lpeak=200, Lblack=0.5979 [cd/m^2], (standard_4k)"`
This is to ensure that you provide enough details to reproduce your results. 

### Predicted quality scores

ColourVideoVDP reports image/video quality in the JOD (Just-Objectionable-Difference) units. The highest quality (no difference) is reported as 10 and lower values are reported for distorted content. In case of very strong distortion, or when comparing two unrelated images, the quality value can drop below 0. 

The main advantage of JODs is that they (a) should be linearly related to the perceived magnitude of the distortion and (b) the difference of JODs can be interpreted as the preference prediction across the population. For example, if method A produces a video with the quality score of 8 JOD and method B gives the quality score of 9 JOD, it means that 75% of the population will choose method B over A. The plots below show the mapping from the difference between two conditions in JOD units to the probability of selecting the condition with the higher JOD score (black numbers on the left) and the percentage increase in preference (blue numbers on the right).

<table>
  <tr>
    <td>Fine JOD scale</td>
    <td>Coarse JOD scale</td>
  </tr>
  <tr>
    <td><img width="512" src="imgs/fine_jod_scale.png"></img></td>
    <td><img width="512" src="imgs/coarse_jod_scale.png"></img></td>
  </tr>
</table>

### Visualization

In addition to the single-valued quality scored in the JOD units, ColourVideoVDP can generate a heatmap (video or image) and a distogram. There are three types of heatmaps:
* supra-threshold - the difference values between 0 and 3 will be mapped to blue to yellow colors (visualizes large differences)
* threshold - the difference values between 0 and 1 will be mapped to green to red colors (visualizes small differences)
* raw - the difference values between 0 and 10 will be mapped to back to white

The `--distogram` command line argument can be followed by a floating point value. If present, it will be used as the maximum JOD value to use in the visualization. The default is 10.

## Example usage

### Command line interface
The main script to run the model on a set of images or videos is [run_cvvdp.py](pycvvdp/run_cvvdp.py), from which the binary `cvvdp` is created . Run `cvvdp --help` for detailed usage information.

For the first example, we will compare the performance of ColourVideoVDP to the popular Delta E 2000 color metric. A video was degraded with 3 different color artifacts using ffmpeg:
1. Gaussian noise: `ffmpeg ... -vf noise=alls=28.55:allf=t+u ...`
1. Increased saturation: `ffmpeg ... -vf eq=saturation=2.418 ...`
1. Colorbalance: `ffmpeg ... -vf colorbalance=rm=0.3:gm=0.3:bm=0.3:rs=0.15:gs=0.2:bs=0.2`

The magnitude of each degradation was adjusted so that the predicted maximum **Delta E 2000 = 33.5**. As Delta E 2000 is not sensitive to spatial or temporal components of the content, it is unable to distinguish between these cases.

To predict quality with ColourVideoVDP (shown below), run:

```bash
cvvdp --test example_media/structure/ferris-test-*.mp4 --ref example_media/structure/ferris-ref.mp4 --display standard_fhd --heatmap supra-threshold --distogram
```

|Original | ![ferris wheel](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/cvvdp/ferris-ref.gif) | Quality | **TODO:** adjust heatmaps |
| :---: | :---: | :---: | :---: |
| Gaussian noise | ![noise](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/cvvdp/ferris-noise.gif) | DE00 = 33.5  <br /> CVVDP = 8.6842 | ![noise-heatmap](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/cvvdp/heatmaps/ferris-noise-supra.gif) |
| Saturation | ![saturation](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/cvvdp/ferris-sat.gif) | DE00 = 33.5 <br /> CVVDP = 6.8960 | ![sat-heatmap](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/cvvdp/heatmaps/ferris-sat-supra.gif) |
| Colorbalance | ![wb](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/cvvdp/ferris-wb.gif) | DE00 = 33.5 <br /> CVVDP = 7.1682 | ![wb-heatmap](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/html_reports/github_examples/cvvdp/heatmaps/ferris-wb-supra.gif) |


### Low-level Python interface
ColourVideoVDP can also be run through the Python interface by instatiating the `pycvvdp.cvvdp` class.

```python
import pycvvdp

I_ref = pycvvdp.load_image_as_array(...)
I_test = pycvvdp.load_image_as_array(...)

cvvdp = pycvvdp.cvvdp(display_name='standard_4k', heatmap='threshold')
JOD, m_stats = cvvdp.predict( I_test, I_ref, dim_order="HWC" )
```

Below, we show an example comparing ColourVideoVDP to the popular SSIM metric. While SSIM is aware of the structure of the content, it operates on luminance only, and is unable to accurately represent the degradation of a color-based artifact like chroma subsampling.

![chroma_ss](imgs/chroma_ss.png)

More examples can be found in these [example scripts](examples).

## Release notes

* v0.1.0 - Initial internal release.

The detailed list of changes can be found in [ChangeLog.md](ChangeLog.md).
