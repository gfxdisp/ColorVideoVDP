import argparse
import logging
import numpy as np
import os
import pandas as pd
import pycvvdp
import sys
import torch
from tqdm import trange

def main():
    parser = argparse.ArgumentParser('Extract features for cvvdp calibration')
    parser.add_argument('quality-file', help='Path to .csv file containinf quality scores.')
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
    parser.add_argument("--gpu", type=int,  default=0, help="Select which GPU to use (e.g. 0), default is GPU 0. Pass -1 to run on the CPU.")
    parser.add_argument('--cache-loc', type=str, default=None, help='Cache the dataset at the provided location using memory mapping.')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume running the metric (skip the conditions that have been already processed).')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    args = parser.parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=level)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu))
    else:
        device = torch.device('cpu')

    if not args.config_dir is None:
        pycvvdp.utils.config_files.set_config_dir(args.config_dir)
        pfile = os.path.join(args.config_dir, "cvvdp_parameters.json")
        if os.path.isfile( pfile ):
            logging.info( f"Using metric parameter file {pfile}")
        else:
            logging.error( f"Cannot find the parameter file {pfile}")
            sys.exit(-1)

    if args.masking == 'base' and args.pooling == 'base':
        metric = pycvvdp.cvvdp(quiet=True, device=device, temp_padding='replicate')
    else:
        metric = pycvvdp.cvvdp_nn(quiet=True, device=device, temp_padding='replicate', masking=args.masking, pooling=args.pooling, ckpt=args.ckpt)

    if not args.worker is None:
        kn = args.worker.split('/',1)
        workerK = int(kn[0])
        workerN = int(kn[1])
        logging.info( f"Worker {workerK} out of {workerN} workers.")

    assert os.path.isfile(args.quality_file), f'Quality file not found at: {args.quality_file}'
    quality_table = pd.read_csv(args.quality_file)
    assert args.split_column in quality_table.columns, f'Split column "{args.split_column}" not found'
    np.random.seed(args.seed)
    unique_cond = np.random.permute(quality_table[args.split_column])
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

        id = args.id_column if args.id_column is None else os.split_ext(test)[0].replace('/', '_')
        split = 'train' if cond in train_cond else 'test'
        dest_name = os.path.join(ft_path, split, id + '_fmap.json')
        if args.resume and os.path.isfile(dest_name):
            logging.info(f'Skipping condition {id}')
            continue

        # TODO: update display model
        # metric.set_display_model(display_photometry=disp_photo, display_geometry=disp_geom)
        # metric.foveated = foveated

        # Create the video source
        vs = pycvvdp.video_source_file(test, ref,
                                    #    display_photometry=disp_photo,
                                    #    resize_resolution=disp_geom.resolution,
                                       verbose=args.verbose)

        # Run the metric
        try:
            with torch.no_grad():
                _, stats = metric.predict_video_source(vs)
        except:
            logging.error( f'Failed on condition {id}' )
            raise

        metric.write_features_to_json(stats, dest_name)

if __name__ == '__main__':
    main()
