from torch.utils.data import Dataset, DataLoader
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
from ifxAvian import Avian
import os

from src.utils.DBF import DBF
from src.utils.doppler_avian import DopplerAlgo
from src.utils.doppler import DopplerAlgo as DopplerAlgoOld
from src.AvianRDKWrapper.ifxRadarSDK import *
from src.utils.common import do_inference_processing, do_inference_processing_RAM
from src.utils.debouncer_time import DebouncerTime


class RadarGestureDataset(Dataset):
    def __init__(self, root_dir='data/recording', annotation_csv='annotation', load_angle=False):
        """
        Args:
            root_dir (str): Base directory containing subfolders of corresponding classes: 'pull', 'push', 'nothing'
            classes (list or None): Optional list of class names (subfolder names). If None, autodetects.
            transform (callable or None): Optional transform to apply to each sample
        """
        # Configuration for the dataset
        self.observation_length = 10
        self.data_root = root_dir
        self.load_angle = load_angle

        # For transformations to time domain

        self.device = Device()
        self.num_rx_antennas = self.device.get_sensor_information()["num_rx_antennas"]
        if self.load_angle:
            self.num_beams = 32
            self.max_angle_degrees = 40
            self.dbf = DBF(self.num_rx_antennas, num_beams = self.num_beams, max_angle_degrees =self.max_angle_degrees)

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
        self.algo = DopplerAlgoOld(self.device.get_config(), self.num_rx_antennas)

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
        self.mapping_idx =  {g: i for i, g in enumerate(gestures)}
        self.idx_to_class = {v: k for k, v in self.mapping_idx.items()}

        return self.mapping_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        self.debouncer = DebouncerTime(memory_length=self.observation_length,)
        sample = self.samples[idx]
        file_path = sample['file_path']
        start = sample['start_frame']
        label = sample['label']

        data = np.load(file_path).astype(np.float32)  # Load the .npy file

        frames = data[start : start + self.observation_length]
        frames = torch.from_numpy(frames)

        for i in range(frames.shape[0]):
            rtm, dtm = self.project_to_time(frames[i])

        rtm = torch.cat(rtm, dim=1)  
        dtm = torch.cat(dtm, dim=1) 

        return rtm, dtm, label

    def get_class_name(self, label):
        return self.idx_to_class[label] 
    
    def get_mapping(self):
        return self.idx_to_class
    
    def project_to_time(self, frame):
        data_all_antennas = []
        for i_ant in range(self.num_rx_antennas):
            mat = frame[i_ant, :, :].squeeze(0).numpy()  
            dfft_dbfs = self.algo.compute_doppler_map(mat, i_ant)
            data_all_antennas.append(dfft_dbfs)

        # Range-Doppler-Map
        range_doppler = do_inference_processing(data_all_antennas)
        self.debouncer.add_scan(range_doppler)
        dtm, rtm = self.debouncer.get_scans()

        # Range-Angle-Map
        if self.load_angle:
            # Rearrange data for DBF
            data_all_antennas_np = np.stack(data_all_antennas, axis=0)
            data_all_antennas_np = data_all_antennas_np.transpose(1,2,0)

            num_chirp_per_frame = data_all_antennas_np.shape[1]/2
            num_samples_per_chirp = data_all_antennas_np.shape[0]

            rd_beam_formed = self.dbf.run(data_all_antennas_np)

            beam_range_energy = np.zeros((num_samples_per_chirp, self.num_beams))
            for i_beam in range(self.num_beams):
                doppler_i = rd_beam_formed[:,:,i_beam]
                beam_range_energy[:,i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(self.num_beams)

            # Maximum energy in Range-Angle map
            max_energy = np.max(beam_range_energy)

            # Rescale map to better capture the peak The rescaling is done in a
            # way such that the maximum always has the same value, independent
            # on the original input peak. A proper peak search can greatly
            # improve this algorithm.
            scale = 150
            beam_range_energy = scale*(beam_range_energy/max_energy - 1)

            # Find dominant angle of target
            _, idx = np.unravel_index(beam_range_energy.argmax(), beam_range_energy.shape)
            angle_degrees = np.linspace(-self.max_angle_degrees, self.max_angle_degrees, self.num_beams)[idx]

            # And plot...
            # self.plot.draw(beam_range_energy, f"Range-Angle map using DBF, angle={angle_degrees:+02.0f} degrees")
            self.ra_queue.put((beam_range_energy.copy(), f"Range-Angle map using DBF, angle={angle_degrees:+02.0f} degrees"))

        else:
            atm = None
    
        return rtm, dtm


class IFXRadarDataset(Dataset):
    def __init__(self, radar_config, root_dir='data/recording', cache_size=3):
        self.file_paths = glob.glob(os.path.join(root_dir, '*'))

        self.idx_mapping = []  # (file_idx, local_idx in the file)
        self.cache_size = cache_size

        for i in range(len(self.file_paths)):
            data = np.load(self.file_paths[i], mmap_mode='r')
            length = len(data['inputs'])
            self.idx_mapping.extend([(i, j) for j in range(length)])

        self._cache = OrderedDict()

        # Radar configuration
        self.radar_config = radar_config
        self.num_rx_antennas = radar_config['num_rx_antennas']
        self.num_beams = radar_config['num_beams']
        self.doppler = DopplerAlgoOld(self.radar_config['dev_config'], 
                                   self.num_rx_antennas)
        
        self.dbf = DBF(self.num_rx_antennas, 
                       num_beams = self.num_beams, 
                       max_angle_degrees = radar_config['max_angle_degrees'])

    def __len__(self):
        return len(self.idx_mapping)
    
    def __getitem__(self, idx):
        file_idx, local_idx = self.idx_mapping[idx]
        file_path = self.file_paths[file_idx]

        if file_idx in self._cache:
            data = self._cache[file_idx]
            self._cache.move_to_end(file_idx)
        else:
            data = np.load(file_path, mmap_mode='r')
            self._cache[file_idx] = data
            self._cache.move_to_end(file_idx)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        frames = data['inputs'][local_idx]  # Frame data
        targets = data['targets'][local_idx]    # TODO: consider how to use targets

        # Process the frames to get RTM, DTM, and ATM
        self.debouncer = DebouncerTime(memory_length=frames.shape[0])
        for i in range(frames.shape[0]):
            rtm, dtm, atm = self.project_to_time(frames[i])

        rtm = torch.cat(rtm, dim=1)
        dtm = torch.cat(dtm, dim=1)
        atm = torch.stack(atm, dim=1) if atm is not None else None
        inputs = torch.stack([rtm, dtm, atm], dim=0)  # Shape: (3, H, W)

        # Process targets into labels
        # take the non zero element
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

        self.debouncer.add_scan(range_doppler, range_angle)
        rtm, dtm, atm = self.debouncer.get_scans()

        return rtm, dtm, atm


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
        batch = {}
        rtm_tensor = torch.stack(rtms)
        dtm_tensor = torch.stack(dtms)

        rdtm_tensor = torch.stack([rtm_tensor, dtm_tensor], dim=1)
        classes_np = np.array(classes)
        classes_tensor = torch.from_numpy(classes_np).long()

        batch['rdtm'] = rdtm_tensor
        batch['class'] = classes_tensor

        return batch

    def get_loader(self):
        return self.dataloader


class IFXDataGen:
    def __init__(self, dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
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
        inputs, labels = zip(*batch)
        inputs_tensor = torch.stack(inputs, dim=0)  # Shape: (batch_size, 3, H, W)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        batch = {
            'inputs': inputs_tensor,
            'labels': labels_tensor
        }
        return batch

    def get_loader(self):
        return self.dataloader


if __name__ == "__main__":
    # dataset = RadarGestureDataset(root_dir='data/recording', annotation_csv='annotation')

    # rtm, dtm, sample_class = dataset[20]

    # # plot_rtm_dtm(rtm, dtm)

    # # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    # dataloader = DataGenerator(dataset, batch_size=8, shuffle=True, max_length=100).get_loader()
    # for batch in dataloader:
    #     batch_videos = batch['rdtm']
    #     batch_classes = batch['class']
    #     print(f"Batch video shape: {batch_videos.shape}")
    #     print(f"Batch class IDs: {batch_classes}")

    dev_config = Avian.DeviceConfig(
        sample_rate_Hz = 2000000,       # 1MHZ
        rx_mask = 7,                      # activate RX1 and RX3
        tx_mask = 1,                      # activate TX1
        if_gain_dB = 25,                  # gain of 33dB
        tx_power_level = 31,              # TX power level of 31
        start_frequency_Hz = 58.5e9,        # 60GHz 
        end_frequency_Hz = 62.5e9,        # 61.5GHz
        num_chirps_per_frame = 32,       # 128 chirps per frame
        num_samples_per_chirp = 64,       # 64 samples per chirp
        chirp_repetition_time_s = 0.0003, # 0.5ms
        frame_repetition_time_s = 1/33,   # 0.15s, frame_Rate = 6.667Hz
        mimo_mode = 'off'                 # MIMO disabled
    )

    config = {'dev_config': dev_config, 
              'num_rx_antennas': 3, 
              'num_beams': 32,
              'max_angle_degrees': 40}
            
    dataset = IFXRadarDataset(config, root_dir='/home/swadiryus/projects/dataset/radar_gesture_dataset')
    dataloader = IFXDataGen(dataset, batch_size=8, shuffle=True, num_workers=0).get_loader()
    for batch in dataloader:
        inputs = batch['inputs']
        labels = batch['labels']
        print(f"Batch inputs shape: {inputs.shape}")
        print(f"Batch labels: {labels}")

    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    # for inputs, targets in dataloader:
    #     print(f"Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
    #     print(f"Unique targets: {np.unique(targets.numpy())}")
    #     break  # Just to test the first batch