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
    def __init__(self, radar_config, root_dir='data/recording', cache_size=3):
        self.file_paths = glob.glob(os.path.join(root_dir, '*'))

        self.idx_mapping = []  # (file_idx, local_idx in the file)
        self.cache_size = cache_size

        for i in range(len(self.file_paths)):
            with np.load(self.file_paths[i], mmap_mode='r') as data:
                shape = data['inputs'].shape
                length = shape[0]
            self.idx_mapping.extend([(i, j) for j in range(length)])

        self._cache = OrderedDict()

        # Radar configuration
        self.radar_config = radar_config
        self.num_rx_antennas = radar_config['num_rx_antennas']
        self.num_beams = radar_config['num_beams']
        self.doppler = DopplerAlgo(self.radar_config['dev_config'], 
                                   self.num_rx_antennas)
        
        self.dbf = DBF(self.num_rx_antennas, 
                       num_beams = self.num_beams, 
                       max_angle_degrees = radar_config['max_angle_degrees'])
        
    def __len__(self):
        return len(self.idx_mapping)
    
    def __getitem__(self, idx):
        file_idx, local_idx = self.idx_mapping[idx]
        file_path = self.file_paths[file_idx]

        with np.load(file_path, mmap_mode='r') as data:
            frames = data['inputs'][local_idx]  # Frame data
            targets = data['targets'][local_idx]    # TODO: consider how to use targets

        # Process the frames to get RTM, DTM, and ATM
        rtm_list, dtm_list, atm_list = [], [], []
        for i in range(frames.shape[0]):
            rtm, dtm, atm = self.project_to_time(frames[i])
            rtm_list.append(rtm) 
            dtm_list.append(dtm)
            atm_list.append(atm)

        rtm = torch.cat(rtm_list, dim=1)
        dtm = torch.cat(dtm_list, dim=1)
        atm = torch.stack(atm_list, dim=1) 
        inputs = torch.stack([rtm, dtm, atm], dim=0)  # Shape: (3, H, W)

        # Process targets into labels
        targets = torch.from_numpy(targets).long()
        labels = targets[targets != 0].unique().item()
        labels = self.map_label_to_contiguous(labels)

        return inputs, labels
    
    def map_label_to_contiguous(self, label):
        # 1,2,3,6,7 to 0,1,2,3,4
        mapping = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
        return mapping.get(label, -1)  # Return -1 if label not found

    def get_class_name(self, label):
        pass

    def get_mapping(self):
        pass

    def project_to_time(self, frame):
        # Range Doppler Map (RDM)
        data_all_antennas = []
        for i in range(self.num_rx_antennas):
            mat = frame[i, :, :]
            dfft_dbfs = self.doppler.compute_doppler_map(mat, i)
            data_all_antennas.append(dfft_dbfs)
        range_doppler = do_inference_processing(data_all_antennas)

        # Range-Angle Map (RAM)
        data_all_antennas_np = np.stack(data_all_antennas, axis=0)
        data_all_antennas_np = data_all_antennas_np.transpose(1,2,0)
        num_samples_per_chirp = data_all_antennas_np.shape[0]

        rd_beam_formed = self.dbf.run(data_all_antennas_np)

        beam_range_energy = np.zeros((num_samples_per_chirp, self.num_beams))
        for i_beam in range(self.num_beams):
            doppler_i = rd_beam_formed[:,:,i_beam]
            beam_range_energy[:,i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(self.num_beams)

        max_energy = np.max(beam_range_energy)
        scale = 150
        beam_range_energy = scale*(beam_range_energy/max_energy - 1)
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
