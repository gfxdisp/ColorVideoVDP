# Calibrate ColourVideoVDP on a new dataset

## Prerequisites
Install additional python libraries:
```bash
pip install torchmetrics
```

## Step 1: Prepare quality CSV
Create a `.csv` file with the following mandatory columns:
1. `test` - test filename
2. `ref` - reference filename
3. `jod` - ground-truth quality in JOD units (typically obtained by running a subjective experiment).

A few optional columns may also be included:
1. `display` - per-condition display model; use this if each row requires a separate display model (photometric or geometric)

**[Header]**: The file may include an optional header containing argparse options. These options will be parsed by our script and will supersede default as well as CLI arguments. See [this example file](calibration/xr-david.csv) for more details.

## Step 2: Extract features
Run the script `extract_features.py` with a single mandatory argument - the `.csv` file from the previous step. Additional arguments may be passed directly or included in the **Header** of the `.csv` file. Run `-h/--help` to obtain descriptions for all arguments.

## Step 3: Run training
Run the script `train.py`, again with a single mandatory argument - the same `.csv` file used to extract features. The result of calibration is a new configuration JSON file provided by argument `--output_file` (default value is "new_parameters.json"). Replace the [original file](pycvvdp/vvdp_data/cvvdp_parameters.json) with this file to use the calibrated parameters.

Intermediate losses and validation metrics are stored at "logs/" by default, update this location by passing `--log-dir`. All logs may be viewed by running a [tensorboard server](https://www.tensorflow.org/tensorboard) as follows:
```bash
tensorboard --logdir logs
```
