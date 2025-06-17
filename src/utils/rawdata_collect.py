from src.internal.fft_spectrum import *
from src.AvianRDKWrapper.ifxRadarSDK import *
from src.utils.doppler import DopplerAlgo
from src.utils.common import do_inference_processing, do_preprocessing
from src.utils.debouncer_time import DebouncerTime
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import threading
import numpy as np
from src.model.simple_model import SimpleCNN
import os
import pandas as pd
import traceback
import time

class DataRecord:
    def __init__(self, save_dir):
        self.debouncer = DebouncerTime()
        self.prev_rtm_tensor = None

        self.time_per_frame = 0.1   # how long is one frame in second
        self.num_frames = 700   # total number of recorded frames 

        self.recording_type = 'push'    # class type

        self.save_dir = os.path.join(save_dir,f'{self.recording_type}/')
        self.save_metric = os.path.join(save_dir, f'metric/{self.recording_type}/')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_metric, exist_ok=True)

        num_el = len([f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))])
        
        self.save_idx = num_el + 1

        print("Press Enter to Start")

    def run(self):
        with Device() as device:
            num_rx_antennas = device.get_sensor_information()["num_rx_antennas"]
            rx_mask = (1 << num_rx_antennas) - 1

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

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                print(f"Created saving directory: {self.save_dir}")

            recorded_frames = []

            for i in range(30):
                print("======= WARMING UP =======")
                start_time = time.time()
                frame_data = device.get_next_frame()
                end_time = time.time()
                elapsed_time = end_time - start_time
                sleep_time = max(0, self.time_per_frame - elapsed_time)
                time.sleep(sleep_time)
            
            print(f"Recording {self.recording_type} {i+1}: Starting in")
            for t in range(3,0,-1):
                print(f"{t}...", end="", flush=True)
                time.sleep(1)
            print("\n ======= STARTING RECORDING =======")

            for i in range(self.num_frames):
                if i % 10 == 0:
                    print(f"Seconds ({self.recording_type}): {i * self.time_per_frame:.1f} s")

                start_time = time.time()

                frame_data = device.get_next_frame()

                # Append each frame to the list
                recorded_frames.append(frame_data)

                end_time = time.time()
                elapsed_time = end_time - start_time
                sleep_time = max(0, self.time_per_frame - elapsed_time)
                time.sleep(sleep_time)

            # Stack and save entire recording
            recorded_frames = np.stack(recorded_frames, axis=0)

            file_name = os.path.join(self.save_dir, f"record_{self.recording_type}_{self.save_idx:04d}.npy")
            np.save(file_name, recorded_frames)

            # Also save metric config
            metric_path = os.path.join(self.save_metric, f"metric_{self.recording_type}_{self.save_idx:04d}.csv")
            pd.DataFrame([metric]).to_csv(metric_path, index=False)

            print(f"Recording saved to {file_name}")
            print("FINISHED")


if __name__ == "__main__":
    datarecord = DataRecord(save_dir=f'data/recording/')

    datarecord.run()
    print("====== Recording is finished ======")