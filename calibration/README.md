# Calibrate ColourVideoVDP on a new dataset

## Prerequisites
Install additional python libraries:
```bash
pip install torchmetrics
```

## Step 1: Prepare quality CSV
Create a `.csv` file with the following mandatory columns:
1. `test` - test filename
2. `reference` - reference filename
3. `jod` - ground-truth quality in JOD units (typically obtained by running a subjective experiment).

A few optional columns may also be included:
- `display` - per-condition display model; use this if each row requires a separate display model (photometric or geometric). The display name should be listed in "display_models.json" located either at the [default](../pycvvdp/vvdp_data) location or at a custom location (supplied by option `--config-dir`).
- `path-prefix` - path to the folder which contains all test and reference videos. If this argument is passed, each row of the quality file should contain relative paths (w.r.t to the provided prefix)

**[Header]**: The file may include an optional header containing argparse options. These options will be parsed by our python scripts and **will supersede** default as well as CLI arguments. See [this example file](xr-david.csv) for more details. Note that short forms are not allowed here, please use full argument names.

## Step 2: Extract features
Run the script `extract_features.py` with a single mandatory argument - the `.csv` file from the previous step. Additional arguments may be passed directly or included in the **Header** of the `.csv` file. Run `-h/--help` to obtain descriptions for all arguments.

Extracted features will be stored as `features/*.json` files by default. If `-f/--features-suffix` is passed (either through CLI or in the `.csv` file), the files will be named `features_{suffix}/*.json`.

## Step 3: Run training
Run the script `train.py`, again with a single mandatory argument - the same `.csv` file used to extract features. The result of calibration is a new configuration JSON file stored at "new_config/cvvdp_parameters.json". The output directory can be changed by passing a custom location using the argument `-o/--output-file`. To use calibrated parameters, pass this directory to `run_cvvdp.py` (or the `cvvdp` executable) using `--config-dir`.

Intermediate losses and validation metrics are stored at "logs/" by default, update this location by passing `--log-dir`. All logs may be viewed by running a [tensorboard server](https://www.tensorflow.org/tensorboard) as follows:
```bash
tensorboard --logdir logs
```

**Important:** It is recommended to monitor the training logs to determine number of epochs needed (specific to each dataset). Training should be stopped when the validation metrics saturate/decrease in performance. Alternatively, save the parameters of the best-performing model by passing `--save best-{val_metric}` where "val_metric" is one of \["rmse", "pearson", "spearman"\].

## Detailed explanations for some arguments
- `-s/--split-column`: The dataset needs to be divided into train and test splits. To test generalization capabilities of the metric, it is preferable to use different scenes or different distortions in the train and test splits. This argument controls how to split the existing dataset, i.e, along which column of the quality file. E.g., if a dataset contains 5 distortions (A, B, C, D, E) and `-s distortion -r 80` is passed, 80% of the distortions (chosen at random) are selected for training (A, B, C, E) and the remaining 20% for testing. Thus, all rows in the quality file with distortions (A, B, C, E) become train images/videos, while rows with distortion C become test images/videos.
- `--seed`: When given "split_column" and "train_ratio", the seed controls the exact split of train and test images/videos. This is useful for reproducable pseudo-random generation of splits.
- `-d/--display`: There are two ways of passing ColourVideoVDP display models (both photometric and geometric). If all test conditions (rows in the quality file) share the same display models, please supply the name of the display directly, e.g., `-d standard_4k`. Alternatively, if different test conditions require different display models, it is recommended to include a separate column titled "display" in the quality file. This can indicate a separate display for each row. In such cases, pass `-d per-row` as an argument to the python scripts. For both cases (single or per-row display), the provided display name must exist in the file "display_models.json" located either at the [default location](../pycvvdp/vvdp_data) or at a custom location (supplied by option `--config-dir`).
