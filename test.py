import numpy as np
import os

data = np.load('/home/swadiryus/projects/dataset/radar_gesture_dataset/user1_e1.npz')
inputs = data['inputs']
targets = data['targets']

print(inputs.shape, targets.shape)

K = 128
small_inputs = inputs[:K, :, :, :, :]
small_targets = targets[:K, :]

os.makedirs('/home/swadiryus/projects/dataset_debug', exist_ok=True)
np.savez_compressed('/home/swadiryus/projects/dataset_debug/radar_gesture_dataset_small.npz', inputs=small_inputs, targets=small_targets)

# save second part of the dataset
small_inputs_2 = inputs[K:2*K-1, :, :, :, :]
small_targets_2 = targets[K:2*K-1, :]

np.savez_compressed('/home/swadiryus/projects/dataset_debug/radar_gesture_dataset_small_2.npz', inputs=small_inputs_2, targets=small_targets_2)