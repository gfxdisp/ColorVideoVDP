import json
import logging
import numpy as np
import os.path as osp
import re
from scipy.interpolate import interp1d
import torch, torch.utils.data as D

class VideoDataset(D.Dataset):
    log_rho_min = -1
    log_rho_max = 6

    def __init__(self, feature_dir, quality_table, split, resample):
        super().__init__()
        logging.info(f'Loading dataset "{self.__class__.__name__}"')
        self.feature_dir = feature_dir
        assert osp.isdir(self.feature_dir), f'Extracted features not found at: {self.feature_dir}'
        self.quality_table = quality_table
        self.split = split
        self.resample = resample

        # Cache for faster training
        self.Q_per_ch, self.base_rho_band = {}, {}

    def __getitem__(self, index):
        """
        Returns:
            qpc:            quality per channel (cxfxb)
            base_rho_band:  spatial frequency of the base band (smallest frequency)
            quality:        subjective quality (in JOD)
        """
        assert index in range(self.__len__()), f'{index} is out of range, len={self.__len__()}'

        row = self.quality_table.iloc[index]
        test_fname, quality = row[['test', 'jod']]

        if test_fname in self.Q_per_ch:
            qpc, base_rho_band = self.Q_per_ch[test_fname], self.base_rho_band[test_fname]
        else:
            feat_fname = osp.join( self.feature_dir, self.split, f'{test_fname}_fmap.json' )
            assert osp.isfile(feat_fname), f'Features missing for "{test_fname}"'
            with open(feat_fname, "r") as json_file:
                features = json.load(json_file)

            f_keys = set([k for k in features.keys() if re.match('t\d+_b\d+', k)])
            bands = len(set([k.split('_')[1].lstrip('b') for k in f_keys]))
            temp_channels = len(set([k.split('_')[0].lstrip('t') for k in f_keys]))
            frames = len(features['t0_b0'])

            if max(features['rho_band']) < 2**self.log_rho_max:
                features['rho_band'].insert(0, 2**self.log_rho_max)
                extrapolate = True
            rho_band = np.array( features['rho_band'] )

            # Resample the features at frequencies [0.5, 1, 2, ..., 64]
            resampled_bands = self.log_rho_max - self.log_rho_min + 2   # Extra entry at the end to store base band
            resampled_qpc = np.empty( (temp_channels,frames,resampled_bands), dtype=np.float32)
            qpc = np.empty( (temp_channels,frames,bands), dtype=np.float32)
            for cc in range(temp_channels):
                for bb in range(bands):
                    qpc[cc,:,bb] = np.array( features[f't{cc}_b{bb}'] ).reshape( (1,1,frames) )

                if self.resample:
                    for tt in range(frames):
                        lut = interp1d(rho_band, np.insert(qpc[cc,tt], 0, 0) if extrapolate else qpc[cc,tt])
                        resampled_qpc[cc,tt] = np.append(lut(2**np.linspace(self.log_rho_max, self.log_rho_min, resampled_bands-1)), qpc[cc,tt,-1])

            if self.resample:
                qpc = torch.tensor(resampled_qpc)

            base_rho_band = np.float32(rho_band[-1])
            self.Q_per_ch[test_fname] = qpc
            self.base_rho_band[test_fname] = base_rho_band

        return qpc, base_rho_band, quality

    def __len__(self):
        return len(self.quality_table)


def collate(batch):
    # Custom collate is needed for unequal batches
    qpc, rho_band, q = [], [], []
    for item in batch:
        qpc.append(torch.tensor(item[0]))
        rho_band.append(item[1])
        q.append(item[2])
    return qpc, torch.tensor(rho_band), torch.tensor(q, dtype=torch.float32)


def get_loaders(feature_dir, train_table, test_table, resample, batch, num_workers):
    train_dataset = VideoDataset(feature_dir, train_table, 'train', resample)
    train_loader = D.DataLoader(train_dataset, batch, num_workers=num_workers, shuffle=True, persistent_workers=True, collate_fn=collate)

    test_dataset = VideoDataset(feature_dir, test_table, 'test', resample)
    test_loader = D.DataLoader(test_dataset, batch, num_workers=num_workers, shuffle=False, persistent_workers=True, collate_fn=collate)

    return train_loader, test_loader
