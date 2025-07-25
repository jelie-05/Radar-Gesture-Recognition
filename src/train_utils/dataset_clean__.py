from torch.utils.data import Dataset, DataLoader
import torch
import glob
import numpy as np
import pandas as pd
from collections import OrderedDict
import os

from src.utils.DBF import DBF
from src.utils.doppler import DopplerAlgo
from src.utils.common import do_inference_processing, do_inference_processing_RAM
from src.utils.debouncer_time import DebouncerTime


class IFXRadarDataset(Dataset):
    def __init__(self, radar_config, file_paths=None, root_dir='data/recording', cache_size=3, device='cpu'):
        self.file_paths = file_paths or glob.glob(os.path.join(root_dir, '*.npz'))
        self.cache_size = cache_size
        self._cache = OrderedDict()
        self.idx_mapping = []

        for file_idx, path in enumerate(self.file_paths):
            with np.load(path, mmap_mode='r') as data:
                length = len(data['inputs'])
            self.idx_mapping.extend([(file_idx, local_idx) for local_idx in range(length)])

        self.radar_config = radar_config
        self.num_rx_antennas = radar_config['num_rx_antennas']
        self.num_beams = radar_config['num_beams']

        self.doppler = DopplerAlgo(radar_config['dev_config'], self.num_rx_antennas)
        self.dbf = DBF(self.num_rx_antennas, self.num_beams, radar_config['max_angle_degrees'])

    def __len__(self):
        return len(self.idx_mapping)

    def _get_data(self, file_idx):
        if file_idx in self._cache:
            return self._cache[file_idx]
        
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)

        data = np.load(self.file_paths[file_idx], mmap_mode='r')
        self._cache[file_idx] = data
        return data

    def __getitem__(self, idx):
        file_idx, local_idx = self.idx_mapping[idx]
        data = self._get_data(file_idx)

        frames = data['inputs'][local_idx]
        targets = data['targets'][local_idx]

        rtm_list, dtm_list, atm_list = [], [], []
        for i in range(frames.shape[0]):
            rtm, dtm, atm = self.project_to_time(frames[i])
            rtm_list.append(rtm)
            dtm_list.append(dtm)
            atm_list.append(atm)

        rtm = torch.cat(rtm_list, dim=1)
        dtm = torch.cat(dtm_list, dim=1)
        atm = torch.stack(atm_list, dim=1)

        inputs = torch.stack([rtm, dtm, atm], dim=0) # Shape: (3, H, W)

        # Faster label mapping
        label = self.map_label_to_contiguous(np.max(targets))
        return inputs, label

    def map_label_to_contiguous(self, label):
        mapping = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
        return mapping.get(label, -1)

    def project_to_time(self, frame):
        data_all_antennas = []
        for i in range(self.num_rx_antennas):
            dfft_dbfs = self.doppler.compute_doppler_map(frame[i], i)
            data_all_antennas.append(dfft_dbfs)

        range_doppler = do_inference_processing(data_all_antennas)

        data_np = np.stack(data_all_antennas, axis=0).transpose(1, 2, 0)
        beam_formed = self.dbf.run(data_np)

        num_samples = data_np.shape[0]
        beam_range_energy = np.zeros((num_samples, self.num_beams))
        for i in range(self.num_beams):
            doppler_i = beam_formed[:, :, i]
            beam_range_energy[:, i] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(self.num_beams)

        scale = 150
        beam_range_energy = scale * (beam_range_energy / beam_range_energy.max() - 1)
        range_angle = do_inference_processing_RAM(beam_range_energy)

        # Transform into time
        processed_range_doppler = range_doppler[0, 0, :, :]
        max_value = processed_range_doppler.max()
        h, w = (processed_range_doppler == max_value).nonzero(as_tuple=True)
        h, w = h[0], w[0]

        rtm = processed_range_doppler[:, w].unsqueeze(1)  # Range-Time Map
        dtm = processed_range_doppler[h, :].unsqueeze(1)

        atm = range_angle.max(axis=1).values  # Angle-Time Map
        return rtm, dtm, atm
