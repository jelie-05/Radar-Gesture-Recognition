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
            outputs.append(tensor_data)
            print(f"Data: {tensor_data.shape}")

            # for channel in range(self.num_channels):
            #     ch_data = f[f'ch{channel}'][:]
            #     ch_data = ch_data.reshape(-1, self.resolution[0], self.resolution[1])
            #     tensor_data = torch.from_numpy(ch_data)
            #     outputs.append(tensor_data)

        video = torch.stack(outputs, dim=1).float()
        print(f"Video shape before transform: {video.shape}")
        video = VideoTransform(self.resolution)(video)

        # class_id = self.class_mapper.transform([class_label])
        # class_id = torch.tensor(class_id, dtype=torch.long)
        class_id = label[0]

        rtm = []
        dtm = []

        for t in range(video.size(0)):
            frame = video[t, 0, :, :]
            max_value = frame.max()
            # max_pos = torch.nonzero(frame == max_value, as_tuple=True)
            # i, j = max_pos[0][0], max_pos[1][0]
            h, w = (frame == max_value).nonzero(as_tuple=True)
            h, w = h[0], w[0]

            rtm.append(frame[h, :].unsqueeze(1))  
            dtm.append(frame[:, w].unsqueeze(1))  
            

        rtm = torch.cat(rtm, dim=1)
        dtm = torch.cat(dtm, dim=1)

        return video, class_id, rtm, dtm


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

if __name__ == "__main__":
    dataset = SoliDataset(data_path='data/SoliData/dsp', resolution=(32, 32), num_channels=3)
    sample_video, sample_class, rtm, dtm = dataset[11]

    print(f"Sample video shape: {sample_video.shape}")
    print(f"Sample class ID: {sample_class}")

    plot_rtm_dtm(rtm, dtm)