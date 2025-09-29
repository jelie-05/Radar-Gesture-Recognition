import numpy as np
import argparse
import os
import glob
import gc
from src.utils.doppler import DopplerAlgo
from src.utils.DBF import DBF
from src.utils.common import do_inference_processing, do_inference_processing_RAM
import torch
import json
from pathlib import Path
from tqdm import tqdm


class TimeProject():
    def __init__(self, radar_config, observation_length=30, offset=0.7, noise_offset=0.1, target_frequency=10.0):
        self.radar_config = radar_config
        self.num_rx_antennas = radar_config['num_rx_antennas']
        self.num_beams = radar_config['num_beams']
    
        self.doppler = DopplerAlgo(radar_config['dev_config'], self.num_rx_antennas)
        self.dbf = DBF(self.num_rx_antennas, self.num_beams, radar_config['max_angle_degrees'])

        self.observation_length = observation_length
        self.offset = offset
        self.noise_offset = noise_offset

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
    
    def tag_to_idx_semantic(self, idx):
        mapping = {0: 'background', 1: 'push', 2: 'left', 3: 'right', 6: 'up', 7: 'down'}
        return mapping.get(idx, -1)

    def map_label_to_contiguous(self, idx_ori):
        semantic = self.tag_to_idx_semantic(idx_ori)
        # mapping = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
        mapping = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 7: 5}  # Background is kept as class 0
        idx_contiguous = mapping.get(idx_ori, -1)
        return idx_contiguous, semantic

    def extract_observation_window(self, label_tensor):
        non_zero_idx = torch.nonzero(label_tensor != 0, as_tuple=True)[0]

        if non_zero_idx.numel() == 0:
            raise ValueError("Tensor contains no non-zero elements, only background.")

        start_of_gesture = non_zero_idx[0].item()
        target_position_start = int(self.offset * self.observation_length) - 1

        # Set certain offset between observation start (oldest frame) and first label (starting frame)
        # The largest the offset, the faster should the gesture detected (labeled frames lies in newest frames)
        start_idx = max(0, start_of_gesture - target_position_start)   
        
        # Add random offset/noise to start_idx
        delta = 1
        start_idx += torch.randint(-delta, delta + 1, (1,)).item()
        
        end_idx = start_idx + self.observation_length

        if end_idx > label_tensor.shape[0]:
            end_idx = len(label_tensor)
            start_idx = end_idx - self.observation_length
            # start_idx = max(0, start_idx)

        return start_idx, end_idx

def main():
    """
        This function converts and extracts the IFX Dataset from raw data (.npz) into time maps (.npy).
        It outputs the radar configuration used in recording and new labeling (with semantic).

        Checklist:
            1. Ensure the radar configuration is correct.
            2. Ensure the observation length is set correctly.
                a. Dataset is recorded w/ frequency of 30 Hz, so observation length of 30 corresponds to 1 second.
                b. Stepsize n is used to reduce number of frames (lower frequency). n = 3 reduce the frequency to 10 Hz.
            3. Shifting in observation windows
            4. Randomize "Background" class
    """
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Convert Dataset into transformed .npy format (Range Map or Time Map)")
    parser.add_argument('--data_dir', type=Path, required=True, help='Path to data')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory for saving')
    parser.add_argument('--format', type=str, default='time', help='time or range')
    parser.add_argument('--observation_length', type=int, default=30, help='Length of observation window in frames')
    # parser.add_argument('--target_frequency', type=float, default=10.0, help='Target frequency (Hz) for the radar considering processing time (lower than actual frequency)')
    parser.add_argument('--none_class', action='store_true', help='Include the background class (0) in the output')
    parser.add_argument('--offset', type=float, default=0.7, help='Offset for the start of the observation window')
    parser.add_argument('--stepsize', type=int, default=1, help='Stepsize for reducing the number of frames (e.g., 3 for 10 Hz from 30 Hz)')

    args = parser.parse_args()

    # Initialize folders
    output_dir = args.output_dir
    data_dir = args.data_dir
    observation_length = args.observation_length
    step_size = args.stepsize

    output_inputs_dir = os.path.join(output_dir, 'inputs/')
    os.makedirs(output_inputs_dir, exist_ok=True)

    none_class = args.none_class

    # Rewrite dev config as dictionary
    dev_config = {
        'sample_rate_Hz': 2000000,
        'rx_mask': 3,
        'tx_mask': 1,
        'if_gain_dB': 31,
        'tx_power_level': 31,
        'start_frequency_Hz': 58.5e9,
        'end_frequency_Hz': 62.5e9,
        'num_chirps_per_frame': 32,
        'num_samples_per_chirp': 64,
        'chirp_repetition_time_s': 0.0003,
        'frame_repetition_time_s': 0.03,
        'mimo_mode': 'off'
    }  
    radar_config = {'dev_config': dev_config, 
                    'num_rx_antennas': 3, 
                    'num_beams': 32,
                    'max_angle_degrees': 55}
    
    timeproject = TimeProject(radar_config, 
                             observation_length=observation_length,)

    # Get all .npz files in the data directory
    all_npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))

    # Excluding files with suffix _fast, _slow, _wrist
    npz_files = [
        f for f in all_npz_files
        if not any(suffix in f for suffix in ['_fast', '_slow', '_wrist',]
                                            #   'user10_', 'user11_', 'user12_',
                                            #   'user1_e1', 'user1_e2']
        )
    ]

    # Storing the label and semantic mapping
    labels = set()
    count = 0

    # Extract per recording
    for npz_file in npz_files:
        print(f"Processing {npz_file}...")
        base_name = os.path.splitext(os.path.basename(npz_file))[0]  # e.g., "user1_e1"
        with np.load(npz_file, mmap_mode='r') as data:
            inputs = data['inputs']    # Shape: [n_recordings, ...]
            targets = data['targets']  # Shape: [n_recordings, ...]

            progress_bar = tqdm(range(inputs.shape[0]), desc=f"Processing {base_name}")

            for i in progress_bar:
                """
                Process the raw data into range-map or time-map.
                    1. Find frames in relevant window for observation (remove background)
                    2. Generate range-doppler and range-angle map
                    3. Project into time map
                """
                frames_i = inputs[i]    # Shape: [num_frames, n_antennas, chirps, samples]
                target_i = targets[i]   # Shape: [num_frames,] -> include background 0 and gestures 1-7

                # Map the target to contiguous and store
                target_torch = torch.from_numpy(target_i)
                label_original = torch.max(target_torch)
                label, semantic = timeproject.map_label_to_contiguous(int(label_original))
                labels.add((label, semantic))

                # Extract observation window
                start_idx, end_idx = timeproject.extract_observation_window(target_torch)
                rtm_list, dtm_list, atm_list = [], [], []
                for j in range(start_idx, end_idx, step_size):
                    rtm, dtm, atm = timeproject.project_to_time(frames_i[j])
                    rtm_list.append(rtm)
                    dtm_list.append(dtm)
                    atm_list.append(atm)

                rtm = torch.cat(rtm_list, dim=1)
                dtm = torch.cat(dtm_list, dim=1)
                atm = torch.stack(atm_list, dim=1)

                time_maps = torch.stack([rtm, dtm, atm], dim=0) # Shape: (3, H, W)
                
                idx_str = f"{i+1:05d}"
                label_str = f"{label:03d}"

                # Naming Format: Which File - Which Recording - Which Class
                input_filename = f"{base_name}_{idx_str}_{label_str}.npy"
                np.save(os.path.join(output_inputs_dir, input_filename), time_maps)

                progress_bar.set_postfix({'label': label})

                del rtm, dtm, atm, time_maps

                delta = 10
                rtm_list, dtm_list, atm_list = [], [], []
                if none_class and count < 7000: # Each class has around 5000 samples
                    count += 1
                    # Save the empty class (background) as well
                    if np.random.choice([True, False]):
                        # Case 1: before the gesture
                        end_idx_none_mark = start_idx 
                        start_idx_none = max(0, end_idx_none_mark - observation_length - torch.randint(0, delta+1, (1,)).item())
                        frame_range = range(start_idx_none, start_idx_none + observation_length, step_size)

                    else:
                        # Case 2: after the gesture
                        start_idx_none_mark = end_idx 
                        end_idx_none = min(start_idx_none_mark + observation_length + torch.randint(0, delta+1, (1,)).item(), frames_i.shape[0])
                        frame_range = range(end_idx_none - observation_length, end_idx_none, step_size)

                    for j in frame_range:
                        rtm, dtm, atm = timeproject.project_to_time(frames_i[j])
                        rtm_list.append(rtm)
                        dtm_list.append(dtm)
                        atm_list.append(atm)

                    rtm = torch.cat(rtm_list, dim=1)
                    dtm = torch.cat(dtm_list, dim=1)
                    atm = torch.stack(atm_list, dim=1)

                    time_maps = torch.stack([rtm, dtm, atm], dim=0) # Shape: (3, H, W)
                    
                    idx_str = f"{i+1:05d}"

                    label_none = 0  # Background class
                    labelnone, semantic_none = timeproject.map_label_to_contiguous(int(label_none))
                    labels.add((labelnone, semantic_none))
                    label_str = f"{labelnone:03d}"

                    # Naming Format: Which File - Which Recording - Which Class
                    input_filename = f"{base_name}_{idx_str}_{label_str}.npy"
                    np.save(os.path.join(output_inputs_dir, input_filename), time_maps)

                    del rtm, dtm, atm, time_maps
                    
                del frames_i, target_i
                gc.collect()  # Clear memory after each save

            del inputs, targets
            gc.collect()  # Clear memory after each save

        print(f"Saved inputs for {base_name} to {output_inputs_dir}.")

    # Saving configs and labels as dict in output_dir
    labels_mapping = dict(sorted(labels, key=lambda x:x[0]))

    with open(f'{output_dir}/labels_mapping.json', 'w') as f:
        json.dump(labels_mapping, f, indent=2)

    print(f"Labels mapping saved to {output_dir}/labels_mapping.json")

    with open(f'{output_dir}/radar_config.json', 'w') as f:
        json.dump(radar_config, f, indent=2)

    print(f"Radar configuration saved to {output_dir}/radar_config.json")

if __name__=="__main__":
    main()
