import numpy as np
import torch
from torchvision import transforms
from src.AvianRDKWrapper.ifxRadarSDK import *
from sklearn.preprocessing import MinMaxScaler, normalize
from cv2 import resize, INTER_AREA

def do_preprocessing(range_doppler):
    # Normalizing to [0, 1]

    range_doppler = np.abs(range_doppler)
    for index, channel in enumerate(range_doppler):
        min = np.min(channel)
        max = np.max(channel)
        range_doppler[index] = (channel - min) / (max - min)

    range_doppler = np.transpose(range_doppler, (2, 1, 0))
    range_doppler = resize(range_doppler, dsize=(32, 32), interpolation=INTER_AREA)
    range_doppler = np.transpose(range_doppler, (2, 1, 0))

    return range_doppler

def do_inference_processing(range_doppler: np.array):
    range_doppler = do_preprocessing(range_doppler)

    range_doppler = torch.from_numpy(range_doppler).float()
    range_doppler = torch.unsqueeze(range_doppler, 0)

    return range_doppler

def do_processing_RAM(range_angle: np.array):
    # Normalizing to [0, 1]
    range_angle = np.abs(range_angle)
    for index, channel in enumerate(range_angle):
        min = np.min(channel)
        max = np.max(channel)
        range_angle[index] = (channel - min) / (max - min)

    range_angle = np.transpose(range_angle, (1, 0))
    range_angle = resize(range_angle, dsize=(32, 32), interpolation=INTER_AREA)
    range_angle = np.transpose(range_angle, (1, 0))

    return range_angle

def do_inference_processing_RAM(range_angle: np.array):
    # Do inference processing for a range-angle map (RAM)
    range_angle = do_processing_RAM(range_angle)
    range_angle = torch.from_numpy(range_angle).float()
    # range_angle = torch.unsqueeze(range_angle, 0)

    return range_angle

class VideoTransform(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        h, w = self.size
        L, C, H, W = video.size()
        rescaled_video = torch.FloatTensor(L, C, h, w)

        transform = transforms.Compose([
            transforms.Resize(self.size, antialias=True),
            # transforms.Normalize(0, 1),
        ])

        for l in range(L):
            frame = video[l, :, :, :]
            frame = transform(frame)
            # plt.imshow(frame.permute(1, 2, 0))
            # plt.show()
            rescaled_video[l, :, :, :] = frame

        return rescaled_video


def configure_device(device: Device, num_receivers: int):
    rx_mask = (1 << num_receivers) - 1

    metric = {
        'sample_rate_Hz': 2500000,
        'range_resolution_m': 0.025,
        'max_range_m': 1,
        'max_speed_m_s': 3,
        'speed_resolution_m_s': 0.024,
        'frame_repetition_time_s': 1 / 9.5,
        'center_frequency_Hz': 60_750_000_000,
        'rx_mask': rx_mask,
        'tx_mask': 1,
        'tx_power_level': 31,
        'if_gain_dB': 25,
    }

    cfg = device.metrics_to_config(**metric)
    device.set_config(**cfg)