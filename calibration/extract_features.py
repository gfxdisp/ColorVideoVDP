import argparse
import logging
import numpy as np
import os
import pandas as pd
import pycvvdp
import sys
import torch
from tqdm import trange

def read_args_from_file(args):
    assert os.path.isfile(args.quality_file), f'Quality file not found at: {args.quality_file}'
    with open(args.quality_file) as f:
        lines = f.readlines()
    n = 0
    for line in lines:
        line = line.strip('\n ')
        if line == '' or line.startswith('#'):  # comments
            n += 1
            continue
        if ':' not in line:                     # stop
            break

        key, val = map(str.strip, line.split(':'))
        key = key.replace('-', '_')
        if key in vars(args).keys():
            if val.lower() == 'true':
                sys.argv.append(f'--{key.replace("_", "-")}')
            else:
                sys.argv.extend([f'--{key.replace("_", "-")}', val])
            logging.info(f'Updating {key} to {val}')
        else:
            logging.warning(f'{key} not found in argparse namespace, skipping')
        n += 1
    return n

def get_args():
    parser = argparse.ArgumentParser('Extract features for cvvdp calibration')
    parser.add_argument('quality_file', help='Path to .csv file containinf quality scores.')
    parser.add_argument('-p', '--path-prefix', default='', help='Prefix for each test and reference file')
    parser.add_argument('-s', '--split-column', default='reference', help='Select the column name for train-test split. Must correspond to an existing column. See calibration/README.md for more details.')
    parser.add_argument('-r', '--train-ratio', type=int, choices=range(100), default=80, help='Percentage of data used for training. E.g., "80" refers to a 80-20 train-test split.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducible splits.')
    parser.add_argument('--masking', default='base', choices=['base', 'mlp'], help='Per-frame masking model.')
    parser.add_argument('--pooling', default='base', choices=['base', 'lstm', 'gru'], help='Reduction method used to pool per-frame features.')
    parser.add_argument('--ckpt', default=None, help='PyTorch checkpoint to retrieve weights/parameters.')
    parser.add_argument('-w', '--worker', default=None, type=str, help='WorkerID and the humber of workers in the format k/N, where N is the total number of workers and k=1..N.')
    parser.add_argument('-f', '--features-suffix', default=None, help='suffix to add add to the features diretory name.')
    parser.add_argument('-c', '--config-paths', type=str, nargs='+', default=[], help="One or more paths to configuration files or directories. The main configurations files are `display_models.json`, `color_spaces.json` and `cvvdp_parameters.json`. The file name must start as the name of the original config file.")
    parser.add_argument('-d', '--display', default=None, help='Display name to create photometric and geometric models.')
    parser.add_argument('--gpu', type=int,  default=0, help='Select which GPU to use (e.g. 0), default is GPU 0. Pass -1 to run on the CPU.')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume running the metric (skip the conditions that have been already processed).')
    parser.add_argument('--full-screen-resize', choices=['bilinear', 'bicubic', 'nearest', 'area'], default=None, help="Both test and reference videos will be resized to match the full resolution of the display. Currently works only with videos.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    args = parser.parse_args()

    # Update config from file
    num_skip = read_args_from_file(args)
    args = parser.parse_args()
    quality_table = pd.read_csv(args.quality_file, skiprows=num_skip)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)

    # if not args.config_dir is None:
    #     pycvvdp.utils.config_files.set_config_dir(args.config_dir)

    # Display model checks
    assert args.display is not None, 'Please select a display name. You may select a single display from "pycvvdp/vvdp_data/display_models" or include a column titled "display" and pass "--display per-row".'
    if args.display == 'per-row':
        assert 'display' in quality_table.columns, 'Per-row display selected but cannot find column "display".'

    return args, quality_table

def main():
    args, quality_table = get_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')

    if args.masking == 'base' and args.pooling == 'base':
        metric = pycvvdp.cvvdp(quiet=True, device=device, display_name=args.display, temp_padding='replicate', config_paths=args.config_paths)
    else:
        metric = pycvvdp.cvvdp_nn(quiet=True, device=device, temp_padding='replicate', config_paths=args.config_paths, masking=args.masking, pooling=args.pooling, ckpt=args.ckpt)

    if not args.worker is None:
        kn = args.worker.split('/',1)
        workerK = int(kn[0])
        workerN = int(kn[1])
        logging.info( f"Worker {workerK} out of {workerN} workers.")

    assert args.split_column in quality_table.columns, f'Split column "{args.split_column}" not found'
    np.random.seed(args.seed)
    unique_cond = np.random.permutation(quality_table[args.split_column].unique())
    train_cond = unique_cond[:(len(unique_cond)*args.train_ratio)//100]

    ft_path = 'features' if args.features_suffix is None else 'features_' + args.features_suffix

    if not os.path.isdir(ft_path):
        try:
            os.makedirs(os.path.join(ft_path, 'train'))
            os.makedirs(os.path.join(ft_path, 'test'))
        except:
            pass # Do not fail - other process could have created that dir

    if not args.worker is None:
        rng_start = workerK-1
        rng_step = workerN
    else:
        rng_start = 0
        rng_step = 1

    for kk in trange(rng_start, len(quality_table), rng_step):
        test, ref, cond = quality_table.loc[kk][['test', 'reference', args.split_column]]

        id = os.path.splitext(test)[0].replace('/', '_')    # Unique ID for each row: test filename without extension
        split = 'train' if cond in train_cond else 'test'
        dest_name = os.path.join(ft_path, split, id + '_fmap.json')
        if args.resume and os.path.isfile(dest_name):
            logging.info(f'Skipping condition {id}')
            continue

        # Some datasets may have different display models for each row
        display = quality_table.loc[kk]['display'] if args.display == 'per-row' else args.display
        disp_photo = pycvvdp.vvdp_display_photometry.load(display, config_paths=args.config_paths)
        disp_geom = pycvvdp.vvdp_display_geometry.load(display, config_paths=args.config_paths)
        metric.set_display_model(display_photometry=disp_photo, display_geometry=disp_geom)

        # Create the video source and run the metric
        try:
            vs = pycvvdp.video_source_file(os.path.join(args.path_prefix, test),
                                        os.path.join(args.path_prefix, ref),
                                        display_photometry=disp_photo,
                                        full_screen_resize=args.full_screen_resize,
                                        resize_resolution=disp_geom.resolution,
                                        verbose=args.verbose, 
                                        config_paths=args.config_paths)

            with torch.no_grad():
                _, stats = metric.predict_video_source(vs)
        except:
            logging.error( f'Failed on condition {id}' )
            raise

        metric.write_features_to_json(stats, dest_name)

if __name__ == '__main__':
    main()
