import numpy as np
import os
import glob

data_dir = '/home/swadiryus/projects/dataset/radar_gesture_dataset/'
output_dir = '/home/swadiryus/projects/dataset_npy/'

output_inputs_dir = os.path.join(output_dir, 'inputs/')
output_targets_dir = os.path.join(output_dir, 'targets/')

# Create directories if they do not exist
os.makedirs(output_inputs_dir, exist_ok=True)
os.makedirs(output_targets_dir, exist_ok=True)

npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))

# Extract per recording
for npz_file in npz_files:
    base_name = os.path.splitext(os.path.basename(npz_file))[0]  # e.g., "user1_e1"
    with np.load(npz_file) as data:
        inputs = data['inputs']    # Shape: [n_recordings, ...]
        targets = data['targets']  # Shape: [n_recordings, ...]

        for i in range(inputs.shape[0]):
            idx_str = f"{i+1:05d}"
            input_filename = f"{base_name}_{idx_str}.npy"
            target_filename = f"{base_name}_{idx_str}.npy"

            np.save(os.path.join(output_inputs_dir, input_filename), inputs[i])
            np.save(os.path.join(output_targets_dir, target_filename), targets[i]) 