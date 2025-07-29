import numpy as np
import argparse
import os
import glob
import gc


def main():
    parser = argparse.ArgumentParser(description="Convert Dataset into transformed .npy format (Range Map or Time Map)")

    parser.add_argument('--data_dir', type=str, help='Path to data')
    parser.add_argument('--output_dir', type=str, help='Output directory for saving')
    parser.add_argument('--format', type=str, default='time', help='time or range')

    output_dir = parser.output_dir
    data_dir = parser.data_dir

    output_inputs_dir = os.path.join(output_dir, 'inputs/')
    output_targets_dir = os.path.join(output_dir, 'targets/')

    # Create directories if they do not exist
    os.makedirs(output_inputs_dir, exist_ok=True)
    os.makedirs(output_targets_dir, exist_ok=True)

    all_npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))

    # Excluding files with suffix _fast, _slow, _wrist
    npz_files = [
        f for f in all_npz_files
        if not any(suffix in f for suffix in ['_fast', '_slow', '_wrist'])
    ]

    # Extract per recording
    for npz_file in npz_files:
        print(f"Processing {npz_file}...")
        base_name = os.path.splitext(os.path.basename(npz_file))[0]  # e.g., "user1_e1"
        with np.load(npz_file, mmap_mode='r') as data:
            inputs = data['inputs']    # Shape: [n_recordings, ...]
            targets = data['targets']  # Shape: [n_recordings, ...]

            for i in range(inputs.shape[0]):
                """
                Process the raw data into range-map or time-map.
                    1. Find frames in relevant window for observation (remove background)
                    2. Generate range-doppler and range-angle map
                    3. Project into time map
                """
                frames_i = inputs[i]
                target_i = targets[i]
                
                idx_str = f"{i+1:05d}"
                input_filename = f"{base_name}_{idx_str}.npy"
                target_filename = f"{base_name}_{idx_str}.npy"
                # np.save(os.path.join(output_inputs_dir, input_filename), inputs[i])
                # np.save(os.path.join(output_targets_dir, target_filename), targets[i]) 

            del inputs, targets
            gc.collect()  # Clear memory after each save

        print(f"Saved inputs and targets for {base_name} to {output_inputs_dir} and {output_targets_dir}.")