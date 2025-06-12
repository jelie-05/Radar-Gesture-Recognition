from torch.utils.data import Dataset, DataLoader
import h5py
from sklearn.preprocessing import LabelEncoder
import torch
import torchvision.transforms as transforms
import re
import glob
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),  '..')))
from doppler import DopplerAlgo
from AvianRDKWrapper.ifxRadarSDK import *
from utils.common import do_inference_processing, do_preprocessing
from utils.debouncer_time import DebouncerTime
import pandas as pd

class RadarGestureDataset(Dataset):
    def __init__(self, root_dir='data/recording', annotation_csv='annotation'):
        """
        Args:
            root_dir (str): Base directory containing subfolders of corresponding classes: 'pull', 'push', 'nothing'
            classes (list or None): Optional list of class names (subfolder names). If None, autodetects.
            transform (callable or None): Optional transform to apply to each sample
        """
        # Configuration for the dataset
        self.observation_length = 10
        self.data_root = root_dir


        # For transformations to time domain

        self.device = Device()
        self.num_rx_antennas = self.device.get_sensor_information()["num_rx_antennas"]

        rx_mask = (1 << self.num_rx_antennas) - 1
        metric = {
            'sample_rate_Hz': 2500000,
            'range_resolution_m': 0.025,
            'max_range_m': 1,
            'max_speed_m_s': 3,
            'speed_resolution_m_s': 0.024,
            'frame_repetition_time_s': 1 / 9.5,
            'center_frequency_Hz': 60_750_000_000,
            'rx_mask': rx_mask,
            'tx_mask': 1,
            'tx_power_level': 31,
            'if_gain_dB': 25,
        }

        cfg = self.device.metrics_to_config(**metric)
        self.device.set_config(**cfg)
        self.algo = DopplerAlgo(self.device.get_config(), self.num_rx_antennas)
        self.debouncer = DebouncerTime(memory_length=self.observation_length,)
        


        # Load samples data from .csv files
        self.annotation = pd.read_csv(os.path.join(self.data_root, f'{annotation_csv}.csv'))
        self.samples = []
        self.gesture_idx = self._build_label_mapping()

        for _, row in self.annotation.iterrows():
            file_name = row['file_name']
            gesture = row['gesture']
            start_frames = eval(row['start_frames']) 
            label = self.gesture_idx[gesture]
            
            file_path = os.path.join(self.data_root, gesture, file_name)
            data = np.load(file_path, mmap_mode='r')
            num_frames = data.shape[0]


            for start in start_frames:
                if start + self.observation_length > num_frames:
                    continue
                self.samples.append({
                    'file_path': file_path,
                    'start_frame': start,
                    'label': label,
                })

    def _build_label_mapping(self):
        gestures = sorted(self.annotation['gesture'].unique())
        return {g: i for i, g in enumerate(gestures)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_path = sample['file_path']
        start = sample['start_frame']
        label = sample['label']

        data = np.load(file_path).astype(np.float32)  # Load the .npy file

        frames = data[start : start + self.observation_length]
        frames = torch.from_numpy(frames)

        rtm_list = []
        dtm_list = []

        for i in range(frames.shape[0]):
            rtm, dtm = self.project_to_time(frames[i])
            rtm_list.append(rtm)
            dtm_list.append(dtm)

        rtm = torch.cat(rtm_list, dim=1)
        dtm = torch.cat(dtm_list, dim=1)

        return rtm, dtm, label

    def get_class_name(self, label):
        return self.idx_to_class[label] 
    
    def project_to_time(self, frame):
        data_all_antennas = []
        for i_ant in range(self.num_rx_antennas):
            mat = frame[i_ant, :, :].squeeze(0).numpy()  
            dfft_dbfs = self.algo.compute_doppler_map(mat, i_ant)
            data_all_antennas.append(dfft_dbfs)

        range_doppler = do_inference_processing(data_all_antennas)
        self.debouncer.add_scan(range_doppler)

        dtm, rtm = self.debouncer.get_scans()
    
        return rtm, dtm

def plot_rtm_dtm(rtm, dtm):
    # Define the custom color map (white for max, blue for min)
    custom_cmap = plt.cm.Blues.reversed()  # Reversing the 'Blues' colormap

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plotting RTM with the custom colormap
    im1 = axs[0].imshow(rtm.cpu(), cmap=custom_cmap, aspect='auto', vmin=0, vmax=rtm.max())
    axs[0].set_title('Range-Time Map (RTM)')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Range')
    plt.colorbar(im1, ax=axs[0])

    # Plotting DTM with the custom colormap
    im2 = axs[1].imshow(dtm.cpu(), cmap=custom_cmap, aspect='auto', vmin=0, vmax=dtm.max())
    axs[1].set_title('Doppler-Time Map (DTM)')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Doppler')
    plt.colorbar(im2, ax=axs[1])

    plt.tight_layout()
    plt.show()

class DataGenerator:
    def __init__(self, dataset, batch_size=8, shuffle=True, max_length=100, num_workers=4, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_length = max_length
        self.num_workers = num_workers

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.custom_collate_fn,
            num_workers=self.num_workers,
            drop_last=drop_last
        )

    def custom_collate_fn(self, batch):
        rtms, dtms, classes = zip(*batch)
        rtm_adjusted = []
        dtm_adjusted = []
        batch = {}
        

        rtm_tensor = torch.stack(rtms)
        dtm_tensor = torch.stack(dtms)

        rdtm_tensor = torch.stack([rtm_tensor, dtm_tensor], dim=1)

        classes_np = np.array(classes)

        # classes_tensor = torch.tensor(classes, dtype=torch.long)
        classes_tensor = torch.from_numpy(classes_np).long()

        # return rdtm_tensor, classes_tensor
        batch['rdtm'] = rdtm_tensor
        batch['class'] = classes_tensor

        return batch

    def get_loader(self):
        return self.dataloader


if __name__ == "__main__":
    dataset = RadarGestureDataset(root_dir='data/recording', annotation_csv='annotation')

    rtm, dtm, sample_class = dataset[20]

    plot_rtm_dtm(rtm, dtm)

    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    dataloader = DataGenerator(dataset, batch_size=8, shuffle=True, max_length=100).get_loader()
    for batch in dataloader:
        batch_videos = batch['rdtm']
        batch_classes = batch['class']
        print(f"Batch video shape: {batch_videos.shape}")
        print(f"Batch class IDs: {batch_classes}")