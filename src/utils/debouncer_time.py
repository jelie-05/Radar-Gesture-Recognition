import numpy as np
import torch

class DebouncerTime:

    def __init__(self, detect_threshold=0.6, noise_threshold=0.3, memory_length=30, min_num_detections=3):

        self.detect_threshold = detect_threshold
        self.noise_threshold = noise_threshold
        self.memory_length = memory_length
        self.min_num_detections = min_num_detections

        self.dtm_memory = []
        self.rtm_memory = []
        self.atm_memory = []  # Angle-Time Map

        self.detection_memory = []

    def add_scan(self, rdm, ram=None, channel=0):
        if len(self.dtm_memory) >= self.memory_length:
            self.dtm_memory.pop(0)
            self.rtm_memory.pop(0)

            if ram is not None:
                self.atm_memory.pop(0)

        # Only add the first channel !!!
        processed_frame = rdm[0, channel, :, :]
        max_value = processed_frame.max()

        h, w = (processed_frame == max_value).nonzero(as_tuple=True)
        h, w = h[0], w[0]

        rtm = processed_frame[:, w].unsqueeze(1)  # Range-Time Map
        dtm = processed_frame[h, :].unsqueeze(1)

        self.dtm_memory.append(dtm)
        self.rtm_memory.append(rtm)

        if ram is not None:
            atm = ram.max(dim=1).values  # Angle-Time Map
            self.atm_memory.append(atm)

    def add_scan_np(self, frame, angle_map=None):
        if len(self.dtm_memory) >= self.memory_length:
            self.dtm_memory.pop(0)
            self.rtm_memory.pop(0)

            if angle_map is not None:
                self.atm_memory.pop(0)

        # Only add the first channel !!!
        processed_frame = frame[0, 0, :, :]
        max_value = np.max(processed_frame)

        # h, w = (processed_frame == max_value).nonzero(as_tuple=True)
        h, w = np.where(processed_frame == max_value)
        h, w = h[0], w[0]

        # rtm = processed_frame[:, w].unsqueeze(1)  # Range-Time Map
        # dtm = processed_frame[h, :].unsqueeze(1)
        rtm = processed_frame[:, w].reshape(-1, 1)
        dtm = processed_frame[h, :].reshape(-1, 1)

        self.dtm_memory.append(dtm)
        self.rtm_memory.append(rtm)

        if angle_map is not None:
            atm = angle_map[h,:].unsqueeze(1)  # Angle-Time Map
            self.atm_memory.append(atm)

    def get_scans(self):
        if self.atm_memory:
            return self.dtm_memory, self.rtm_memory, self.atm_memory
        else:
            return self.dtm_memory, self.rtm_memory

    def reset(self):
        self.dtm_memory = []
        self.rtm_memory = []
        self.atm_memory = []
        self.detection_memory = []