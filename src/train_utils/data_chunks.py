import numpy as np
import os
import argparse

def split_npz(npz_path, output_dir, chunk_size=1000):
    os.makedirs(output_dir, exist_ok=True)

    # Load the full .npz file
    print(f"Loading {npz_path} ...")
    data = np.load(npz_path)

    inputs = data['inputs']
    targets = data['targets']
    print(f"Data shape: {inputs.shape}")
    total_samples = inputs.shape[0]

    print(f"Total samples: {total_samples}")
    print(f"Splitting into chunks of size {chunk_size}...")

    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    base_name = os.path.splitext(os.path.basename(npz_path))[0]

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_samples)

        inputs_chunk = inputs[start:end]
        targets_chunk = targets[start:end]

        out_path = os.path.join(output_dir, f"{base_name}_sample_{i:05d}.npz")
        np.savez(out_path, inputs=inputs_chunk, targets=targets_chunk)

        print(f"Saved {out_path} with {end - start} samples")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("npz_path", help="Path to the large .npz file")
    # parser.add_argument("--output_dir", default="split_npz", help="Directory to save smaller .npz files")
    # parser.add_argument("--chunk_size", type=int, default=1000, help="Number of samples per output file")
    # args = parser.parse_args()

    # split_npz(args.npz_path, args.output_dir, args.chunk_size)

    npz_path = '/home/swadiryus/projects/dataset/radar_gesture_dataset/user1_e1.npz'
    output_dir = '/home/swadiryus/projects/dataset/chunks/'
    chunk_size = 100
    split_npz(npz_path, output_dir, chunk_size)