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
import numpy as np

class VideoTransform(object):
    def __init__(self, resolution=(32, 32)):
        self.resolution = resolution

    def __call__(self, video):
        h, w = self.resolution
        L, C, H, W = video.size()
        rescaled_video = torch.FloatTensor(L, C, h, w)

        transform = transforms.Compose([
            transforms.Resize(self.resolution, antialias=True),
            transforms.Normalize(0, 1)
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, class_label = self.data[idx]

        outputs = []

        use_channel = 0

        with h5py.File(video_path, 'r') as f:
            label = f['label'][()]
            data = f['ch{}'.format(use_channel)][()]
            data = data.reshape(-1, self.resolution[0], self.resolution[1])
            tensor_data = torch.from_numpy(data)
            # print(f"data shape: {tensor_data.shape}; max: {tensor_data.max()}, min: {tensor_data.min()}")
            outputs.append(tensor_data)

        video = torch.stack(outputs, dim=1).float()
        video = VideoTransform(self.resolution)(video)

        class_id = label[0]

        rtm = []
        dtm = []
        rdtm = []

        for t in range(video.size(0)):
            frame = video[t, 0, :, :]
            max_value = frame.max()
            # max_pos = torch.nonzero(frame == max_value, as_tuple=True)
            # i, j = max_pos[0][0], max_pos[1][0]
            h, w = (frame == max_value).nonzero(as_tuple=True)
            h, w = h[0], w[0]

            rtm.append(frame[h, :].unsqueeze(1))  
            dtm.append(frame[:, w].unsqueeze(1))  

            rdtm.append(frame[h, :].unsqueeze(1))
            rdtm.append(frame[:, w].unsqueeze(1))  
            

        rtm = torch.cat(rtm, dim=1)
        dtm = torch.cat(dtm, dim=1)
        rdtm = torch.cat(rdtm, dim=1)

        return rtm, class_id, dtm


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
        rtms, classes, dtms = zip(*batch)
        rtm_adjusted = []
        dtm_adjusted = []
        batch = {}

        for video in rtms:
            if video.size(1) < self.max_length:
                padded_video = torch.cat([video, torch.zeros(32, self.max_length - video.size(1))], dim=1)
                rtm_adjusted.append(padded_video)
            else:
                truncated_video = video[:, :self.max_length]
                rtm_adjusted.append(truncated_video)

        for video in dtms:
            if video.size(1) < self.max_length:
                padded_video = torch.cat([video, torch.zeros(32, self.max_length - video.size(1))], dim=1)
                dtm_adjusted.append(padded_video)
            else:
                truncated_video = video[:, :self.max_length]
                dtm_adjusted.append(truncated_video)
        

        rtm_tensor = torch.stack(rtm_adjusted)
        dtm_tensor = torch.stack(dtm_adjusted)

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
    dataset = SoliDataset(data_path='data/SoliData/dsp', resolution=(32, 32), num_channels=3)
    rtm, sample_class, dtm = dataset[20]

    # print(f"Sample video shape: {sample_video.shape}")
    print(f"Sample class ID: {sample_class}")

    plot_rtm_dtm(rtm, dtm)

    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    dataloader = DataGenerator(dataset, batch_size=8, shuffle=True, max_length=100).get_loader()
    for batch_videos, batch_classes in dataloader:
        print(f"Batch video shape: {batch_videos.shape}")
        print(f"Batch class IDs: {batch_classes}")
        break