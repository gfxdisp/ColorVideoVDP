import argparse
import logging
import numpy as np
import os
import pandas as pd
import pycvvdp
import sys
import torch
from tqdm import trange

def get_args():
    parser = argparse.ArgumentParser('Extract features for cvvdp calibration')
    parser.add_argument('quality_file', help='Path to .csv file containinf quality scores.')
    parser.add_argument('-p', '--path-prefix', default='', help='Prefix for each test and reference file')
    parser.add_argument('-e', '--extension', default='mp4', help='Extension to file name')
    parser.add_argument('-s', '--split-column', default='ref', help='Column name for train-test split.')
    parser.add_argument('-r', '--train-ratio', type=int, choices=range(100), default=80, help='Ratio of training split.')
    parser.add_argument('-i', '--id-column', default=None, help='Column name for unique per-row ID.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducible splits.')
    parser.add_argument('--masking', default='base', choices=['base', 'mlp'], help='Per-frame masking model.')
    parser.add_argument('--pooling', default='base', choices=['base', 'lstm', 'gru'], help='Reduction method used to pool per-frame features.')
    parser.add_argument('--ckpt', default=None, help='PyTorch checkpoint to retrieve weights/parameters.')
    parser.add_argument('-w', '--worker', default=None, type=str, help='WorkerID and the humber of workers in the format k/N, where N is the total number of workers and k=1..N.')
    parser.add_argument('-f', '--features-suffix', default=None, help='suffix to add add to the features diretory name.')
    parser.add_argument('-c', '--config-dir', default=None, help='Metric config dir.')
    parser.add_argument('--display-dir', default=None, help='Display model config dir.')
    parser.add_argument('-d', '--display', default=None, help='Display name to create photometric and geometric models.')
    parser.add_argument('--gpu', type=int,  default=0, help='Select which GPU to use (e.g. 0), default is GPU 0. Pass -1 to run on the CPU.')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume running the metric (skip the conditions that have been already processed).')
    parser.add_argument("--full-screen-resize", choices=['bilinear', 'bicubic', 'nearest', 'area'], default=None, help="Both test and reference videos will be resized to match the full resolution of the display. Currently works only with videos.")
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    args = parser.parse_args()

    # Update config from file
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
        if val.lower() == 'true':
            sys.argv.append(f'--{key.replace("_", "-")}')
        else:
            sys.argv.extend([f'--{key.replace("_", "-")}', val])
        logging.info(f'Updating {key} to {val}')
        n += 1

    args = parser.parse_args()
    quality_table = pd.read_csv(args.quality_file, skiprows=n)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)

    if not args.config_dir is None:
        pycvvdp.utils.config_files.set_config_dir(args.config_dir)
        pfile = os.path.join(args.config_dir, "cvvdp_parameters.json")
        if os.path.isfile( pfile ):
            logging.info( f"Using metric parameter file {pfile}")
        else:
            logging.error( f"Cannot find the parameter file {pfile}")
            sys.exit(-1)

    # Display model checks
    assert args.display is not None, 'Please select a display name. You may select a single name or include a column titled "display" and pass "--display per-row".'
    pycvvdp.utils.config_files.set_config_dir(args.display_dir)
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
        metric = pycvvdp.cvvdp(quiet=True, device=device, temp_padding='replicate')
    else:
        metric = pycvvdp.cvvdp_nn(quiet=True, device=device, temp_padding='replicate', masking=args.masking, pooling=args.pooling, ckpt=args.ckpt)

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
        test, ref, cond = quality_table.loc[kk][['test', 'ref', args.split_column]]

        id = os.path.splitext(test)[0].replace('/', '_') if args.id_column is None else args.id_column
        split = 'train' if cond in train_cond else 'test'
        dest_name = os.path.join(ft_path, split, id + '_fmap.json')
        if args.resume and os.path.isfile(dest_name):
            logging.info(f'Skipping condition {id}')
            continue

        # Some datasets may have different display models for each row
        display = quality_table.loc[kk]['display'] if args.display == 'per-row' else args.display
        disp_photo = pycvvdp.vvdp_display_photometry.load(display)
        disp_geom = pycvvdp.vvdp_display_geometry.load(display)
        metric.set_display_model(display_photometry=disp_photo, display_geometry=disp_geom)

        # Create the video source and run the metric
        try:
            vs = pycvvdp.video_source_file(os.path.join(args.path_prefix, f'{test}.{args.extension}'),
                                        os.path.join(args.path_prefix, f'{ref}.{args.extension}'),
                                        display_photometry=disp_photo,
                                        full_screen_resize=args.full_screen_resize,
                                        resize_resolution=disp_geom.resolution,
                                        verbose=args.verbose)

            with torch.no_grad():
                _, stats = metric.predict_video_source(vs)
        except:
            logging.error( f'Failed on condition {id}' )
            raise

        metric.write_features_to_json(stats, dest_name)

if __name__ == '__main__':
    main()
