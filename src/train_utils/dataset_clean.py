from torch.utils.data import Dataset
import torch
import glob
import numpy as np
import json
import os
import math
from src.utils.DBF import DBF
from src.utils.doppler import DopplerAlgo
import torch.nn.functional as F


class IFXRadarDatasetRaw(Dataset):
    def __init__(self, radar_config, root_dir='data/inputs', observation_length=30):
        input_dir = os.path.join(root_dir, 'inputs')
        target_dir = os.path.join(root_dir, 'targets')
        self.input_paths = sorted(glob.glob(os.path.join(input_dir, '*.npy')))
        self.target_paths = sorted(glob.glob(os.path.join(target_dir, '*.npy')))
        assert len(self.input_paths) == len(self.target_paths), "Mismatch in number of input and target files"

        self.radar_config = radar_config
        self.num_rx_antennas = radar_config['num_rx_antennas']
        self.num_beams = radar_config['num_beams']

        self.doppler = DopplerAlgo(radar_config['dev_config'], self.num_rx_antennas)
        self.dbf = DBF(self.num_rx_antennas, self.num_beams, radar_config['max_angle_degrees'])

        self.observation_length = observation_length
        self.offset = 0.7   # 0.6: Gesture starts at 0.6 * observation length

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        frames = np.load(self.input_paths[idx])  # [num_frames: 100, n_antennas, chirps, samples]
        targets = np.load(self.target_paths[idx])

        frames = torch.from_numpy(frames).float()  # Convert to float tensor
        targets = torch.from_numpy(targets).int()  # Convert to int tensor

        # Label mapping
        label_non_contiguous = torch.max(targets)
        label = self.map_label_to_contiguous(int(label_non_contiguous))

        start_idx, end_idx = self.extract_observation_window(targets)

        self.doppler.mti_history.zero_()

        # rtm_list, dtm_list, atm_list = [], [], []
        # for i in range(start_idx, end_idx):
        #     rtm, dtm, atm = self.project_to_time(frames[i])
        #     rtm_list.append(rtm)
        #     dtm_list.append(dtm)
        #     atm_list.append(atm)

        # rtm = torch.cat(rtm_list, dim=1)
        # dtm = torch.cat(dtm_list, dim=1)
        # atm = torch.stack(atm_list, dim=1)

        rtm_one, dtm_one, atm_one = self.project_to_time(frames[start_idx])
        rtm_list = [rtm_one] * (end_idx - start_idx)
        dtm_list = [dtm_one] * (end_idx - start_idx)
        atm_list = [atm_one] * (end_idx - start_idx)
        rtm = torch.cat(rtm_list, dim=1)  # Shape: (32, 32, N)
        dtm = torch.cat(dtm_list, dim=1)  # Shape: (32, 32, N)
        atm = torch.stack(atm_list, dim=1)  # Shape: (32, N)

        inputs = torch.stack([rtm, dtm, atm], dim=0) # Shape: (3, H, W)

        return inputs, label

    def extract_observation_window(self, label_tensor):
        non_zero_idx = torch.nonzero(label_tensor != 0, as_tuple=True)[0]

        if non_zero_idx.numel() == 0:
            raise ValueError("Tensor contains no non-zero elements")
        
        start_of_gesture = non_zero_idx[0].item()
        target_position_start = int(self.offset * self.observation_length) - 1

        start_idx = max(0, start_of_gesture - target_position_start)
        end_idx = start_idx + self.observation_length

        if end_idx > label_tensor.shape[0]:
            end_idx = len(label_tensor)
            start_idx = end_idx - self.observation_length
            start_idx = max(0, start_idx)

        return start_idx, end_idx

    def map_label_to_contiguous(self, label):
        mapping = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
        return mapping.get(label, -1)
    
    def get_class_name(self, label):
        pass

    def get_mapping(self):
        pass

    def project_to_time(self, frame: torch.Tensor):
        assert frame.shape[0] == self.num_rx_antennas, "Mismatch in antenna count"

        # Batched Doppler processing -> (N_ant, N_range, N_doppler)
        doppler_maps = self.doppler.compute_doppler_map(frame)

        # Transpose to (N_range, N_doppler, N_antennas)
        doppler_maps_for_dbf = doppler_maps.permute(1, 2, 0).contiguous()

        # Apply Digital Beamforming -> (N_range, N_doppler, N_beams)
        beam_formed = self.dbf.run(doppler_maps_for_dbf)

        # Compute energy over Doppler axis
        beam_range_energy = torch.linalg.norm(beam_formed, dim=1) / math.sqrt(self.num_beams)  # (N_range, N_beams)

        # Scale and normalize
        scale = 150.0
        beam_range_energy = beam_range_energy / (beam_range_energy.max() + 1e-6)
        beam_range_energy = scale * (beam_range_energy - 1.0)  # shape: (N_range, N_beams)

        # Range-Angle Map processing (-> 32x32)
        range_angle = self.do_inference_processing_RAM(beam_range_energy)  # shape: (32, 32)

        # Range-Doppler preprocessing
        # Convert Doppler maps to (C, H, W)
        range_doppler_input = doppler_maps  # (N_ant, N_range, N_doppler) -> (C, H, W)
        range_doppler = self.do_inference_processing(range_doppler_input)  # (1, C, 32, 32)

        # Extract max activation point in processed RD map
        processed_range_doppler = range_doppler[0].abs().sum(dim=0)  # shape: (32, 32)
        max_value = processed_range_doppler.max()
        h, w = (processed_range_doppler == max_value).nonzero(as_tuple=True)
        h, w = h[0], w[0]

        # Range-Time and Doppler-Time Maps
        rtm = processed_range_doppler[:, w].unsqueeze(1)  # (32, 1)
        dtm = processed_range_doppler[h, :].unsqueeze(1)  # (32, 1)

        # Angle-Time Map (max over range axis)
        atm = range_angle.max(dim=0).values  # (32,)

        return rtm, dtm, atm
    
    def normalize_tensor_per_channel(self, x: torch.Tensor, eps=1e-6):
        """
        Normalize tensor per-channel to [0, 1] along HxW.
        Input shape: (C, H, W)
        """
        min_val = x.view(x.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
        max_val = x.view(x.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
        return (x - min_val) / (max_val - min_val + eps)

    def do_inference_processing(self,range_doppler: torch.Tensor, size=(32, 32)) -> torch.Tensor:
        """
        Normalize, resize, and return Range-Doppler tensor.
        Input:  Tensor of shape (C, H, W) or (H, W, C) as float32 or complex64
        Output: Tensor of shape (1, C, size[0], size[1]) (batch-like)
        """
        if range_doppler.ndim == 3 and range_doppler.shape[0] not in [1, 2, 3, 4, 8]:
            # Convert (H, W, C) → (C, H, W)
            range_doppler = range_doppler.permute(2, 0, 1)

        if torch.is_complex(range_doppler):
            range_doppler = range_doppler.abs()

        range_doppler = self.normalize_tensor_per_channel(range_doppler)

        # Resize to (C, 32, 32)
        range_doppler = F.interpolate(range_doppler.unsqueeze(0), size=size, mode='area')
        return range_doppler  # shape: (1, C, 32, 32)


    def do_inference_processing_RAM(self,range_angle: torch.Tensor, size=(32, 32)) -> torch.Tensor:
        """
        Normalize, resize, and return Range-Angle tensor.
        Input:  Tensor of shape (H, W)
        Output: Tensor of shape (H', W') = (32, 32)
        """
        if torch.is_complex(range_angle):
            range_angle = range_angle.abs()

        # Normalize to [0, 1]
        min_val = range_angle.min()
        max_val = range_angle.max()
        range_angle = (range_angle - min_val) / (max_val - min_val + 1e-6)

        # Resize to (1, 1, 32, 32)
        range_angle = F.interpolate(range_angle.unsqueeze(0).unsqueeze(0), size=size, mode='area')

        return range_angle.squeeze(0).squeeze(0)  # shape: (32, 32)
    

class IFXRadarDataset(Dataset):
    def __init__(self, datadir):
        self.datadir = datadir
        input_dir = os.path.join(datadir, 'inputs/')
        self.input_paths = sorted(glob.glob(os.path.join(input_dir, '*.npy')))

        with open(f'{datadir}/labels_mapping.json', 'r') as f:
            self.labels_mapping = json.load(f)

    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, idx):
        filename = self.input_paths[idx]
        frames = np.load(filename)  
        label = self.extract_idx(filename)

        # Convert to torch
        frames = torch.from_numpy(frames).float()
        
        # reduced and take every 3 frames
        frames = frames[:, :, ::3]

        return frames, label

    def extract_idx(self, filename):
        name_no_ext = filename.replace('.npy', '')
        label_str = name_no_ext.split('_')[-1]
        return int(label_str)

