from torch.utils.data import Dataset, DataLoader
import h5py
from sklearn.preprocessing import LabelEncoder
import torch
import torchvision.transforms as transforms
import re
import glob
import os
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence


class VideoTransform(object):
    def __init__(self, resolution=(32, 32)):
        self.resolution = resolution

    def __call__(self, video):
        h, w= self.resolution
        L, C, H, W = video.size()
        rescaled_video = torch.FloatTensor(L, C, h, w)

        vid_mean = video.mean()
        vid_std = video.std()

        transform = transforms.Compose([
            transforms.Resize(self.resolution, antialias=True),
            transforms.Normalize(0,1)
            # transforms.Normalize(mean=vid_mean, std=vid_std)    # ??? why in original 0,1?
        ])

        for l in range(L):
            frame = video[l]
            frame = transform(frame)
            rescaled_video[l] = frame

        return rescaled_video

class SoliDataset(Dataset):
    def __init__(self, data_path='data/SoliData/dsp', resolution=(32,32), num_channels=3):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.data_path = os.path.join(base_dir, data_path)
        self.resolution = resolution
        self.num_channels = num_channels
        
        self.data = []
        classes = set()

        video_paths = glob.glob(f"{self.data_path}/*.h5")
        if not video_paths:
            raise ValueError(f"No .h5 files found in {self.data_path}. Please check the path.")
        
        for video_path in video_paths:
            class_label = re.findall(r'(\d+)_\d+_\d+.h5', video_path)
            if class_label:
                self.data.append((video_path, class_label[0]))
                classes.add(class_label[0])

        self.class_mapper = LabelEncoder()
        self.class_mapper.fit(list(classes))
        print("Classes found:", self.class_mapper.classes_)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx >= len(self.data) or idx < 0:
            print(f"Invalid index {idx}. Returning None.")
            return None
        
        video_path, class_label = self.data[idx]
        print(f"Loading video from: {video_path}, Label: {class_label}")

        outputs = []

        try:
            with h5py.File(video_path, 'r') as f:
                for channel in range(self.num_channels):
                    print(f"num channels: {self.num_channels}")
                    if f"ch{channel}" not in f:
                        raise KeyError(f"Channel {channel} not found in {video_path}.")
                    
                    ch_data = f[f"ch{channel}"][:]
                    ch_data = ch_data.reshape(-1, self.resolution[0], self.resolution[1])
                    tensor_data = torch.from_numpy(ch_data)
                    outputs.append(tensor_data)

            video = torch.stack(outputs, dim=1).float()
            print("Video shape before transform:", video.shape)

            # Apply transformation
            video = VideoTransform(self.resolution)(video)
            print("Video shape after transform:", video.shape)

            # Convert class label
            class_id = self.class_mapper.transform([class_label])
            class_id = torch.tensor(class_id, dtype=torch.long)
            print("Class ID:", class_id)

            return video, class_id

        except Exception as e:
            print(f"Error loading video: {e}")
            return None

def plot_video_frames(video, num_frames=5):
    """
    Plots the first 'num_frames' frames of the video.
    """
    num_frames = min(num_frames, video.size(0))
    plt.figure(figsize=(15, 5))
    
    for i in range(num_frames):
        frame = video[i]
        plt.subplot(1, num_frames, i + 1)
        plt.imshow(frame[0].cpu(), cmap='gray')
        plt.title(f"Frame {i + 1}")
        plt.axis('off')

    plt.show()

def custom_collate_fn(batch):
    videos, classes = zip(*batch)

    # Find the longest video in the batch
    max_length = max(video.size(0) for video in videos)
    
    padded_videos = []
    for video in videos:
        pad_size = max_length - video.size(0)
        if pad_size > 0:
            # Pad the video with zeros to match the max length
            padding = torch.zeros((pad_size, video.size(1), video.size(2), video.size(3)))
            padded_video = torch.cat([video, padding], dim=0)
        else:
            padded_video = video
        padded_videos.append(padded_video)

    # Stack videos and classes
    padded_videos = torch.stack(padded_videos)
    classes = torch.stack(classes)

    return padded_videos, classes

if __name__ == "__main__":
    dataset = SoliDataset(data_path='data/SoliData/dsp', resolution=(32, 32), num_channels=4)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # Loading a sample to test
    sample_video, sample_class = dataset[5]
    print(f"Sample video shape: {sample_video.shape}")
    print(f"Sample class ID: {sample_class}")

    # Plotting the first few frames of the video
    print("Visualizing the video frames...")
    plot_video_frames(sample_video, num_frames=5)

    # Using DataLoader for batching
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    for batch_videos, batch_classes in dataloader:
        print(f"Batch video shape: {batch_videos.shape}")
        print(f"Batch class IDs: {batch_classes}")
        break