from internal.fft_spectrum import *
from AvianRDKWrapper.ifxRadarSDK import *
from doppler import DopplerAlgo
from utils.common import do_inference_processing, do_preprocessing
from utils.debouncer_time import DebouncerTime
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import threading
import numpy as np
from model.simple_model import SimpleCNN
import os
import pandas as pd
import traceback
import time

class DataRecord:
    def __init__(self, save_dir):
        self.debouncer = DebouncerTime()
        self.prev_rtm_tensor = None

        self.frame_time = 0.1   # how long is one frame in second
        self.num_frames = 15    # number of frames in one recording
        self.num_recordings = 150   # total recording
        self.recording_type = 'pull'    # class type
        self.save_dir = os.path.join(save_dir,f'{self.recording_type}/')
        os.makedirs(self.save_dir, exist_ok=True)

        num_el = len([f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))])
        self.overwrite = True   # Overwrite or extend

        if self.overwrite:
            print("======= starting from zero =======")
            if num_el != 0:
                print(f"OVERWRITING THE FOLDER: {self.recording_type}")
                input("Press enter to continue")
            self.start = 0
        else:
            self.start = num_el
            print(f"======= starting from {self.start:4d} =======")

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

            algo = DopplerAlgo(device.get_config(), num_rx_antennas)      

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                print(f"Created saving directory: {self.save_dir}")

            for i in range(self.num_recordings):
                if i!=0 and i % 50==0:
                    input(">>>>>> Pause :)")
                print(f"Recording {self.recording_type} {i+1}: Starting in")
                for t in range(1,0,-1):
                    print(f"{t}...", end="", flush=True)
                    time.sleep(1)

                print("GO!")
                recorded_frames = []

                counter = 0

                start_time = time.time()
                while len(recorded_frames) <= self.num_frames:
                    start_loop = time.time()
                    counter+=1

                    frame_data = device.get_next_frame()

                    data_all_antennas = []
                    for i_ant in range(num_rx_antennas):
                        mat = frame_data[i_ant, :, :]
                        dfft_dbfs = algo.compute_doppler_map(mat, i_ant)
                        data_all_antennas.append(dfft_dbfs)

                    range_doppler = do_inference_processing(data_all_antennas)

                    channel_0 = range_doppler[:,0,:,:]
                    recorded_frames.append(channel_0.cpu().detach().numpy())

                    end_loop = time.time()
                    elapsed_time = end_loop-start_loop
                    sleep_time = max(0,self.frame_time-elapsed_time)
                    time.sleep(sleep_time)

                end_time = time.time()
                print(f"recording time: {end_time-start_time} s")
                recorded_frames = np.stack(recorded_frames, axis=0)

                j = i + self.start
                file_name = os.path.join(self.save_dir, f"record_{self.recording_type}_{j:04d}.npy")

                np.save(file_name, recorded_frames)
                print("FINISHED")
                time.sleep(0.5)

if __name__ == "__main__":
    datarecord = DataRecord(save_dir=f'data/recording/')

    datarecord.run()
    print("====== Recording is finished ======")