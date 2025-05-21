import numpy as np
import torch
import json

with open("data/pull/BGT60TR13C_record_20250520-144605.json") as f:
    meta = json.load(f)

num_frames = meta["recording"]["frame_count"]
num_chirps = meta["device_config"]["fmcw_single_shape"]["num_chirps_per_frame"]
num_samples = meta["device_config"]["fmcw_single_shape"]["num_samples_per_chirp"]
num_rx = len(meta["device_config"]["fmcw_single_shape"]["rx_antennas"])

dtype = np.float128
raw_data = np.fromfile("data/pull/BGT60TR13C_record_20250520-144605.raw.bin", dtype=dtype)

try:
    radar_cube = raw_data.reshape((num_frames, num_chirps, num_rx, num_samples))
    radar_tensor = torch.from_numpy(radar_cube)
    print("loaded shape:", radar_tensor.shape)
except Exception as e:
    print(f"reshape failed: {e}")