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
from src.train_utils.dataset import RadarGestureDataset, DataGenerator

import time

class PredictionInference:
    def __init__(self, observation_length, num_classes):
        self.num_classes = num_classes
        self.observation_length = observation_length

        self.debouncer = DebouncerTime(memory_length=self.observation_length,)
        self.visualizer = Visualizer(observation_length=self.observation_length, num_classes=num_classes)
        self.prev_rtm_tensor = None

        self.model = SimpleCNN(in_channels=2, num_classes=self.num_classes)
        self.model.eval()
        # model_path = "runs/trained_models/radar_edge_network.pth"
        model_path = 'runs/trained_models/train_0606-last.pth'
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print("[Model] Loaded successfully.")
            input("[Model] Press Enter to continue...")
        else:
            input("[Model] Model file not found. Please ensure the path is correct and the model is trained.")
            self.model = self.model.to('cpu')

        # self.dataset = NumpyDataset()
        self.dataset = RadarGestureDataset(root_dir='data/recording', annotation_csv='annotation')
        self.datagen = DataGenerator(self.dataset, batch_size=1, shuffle=False, max_length=self.observation_length, num_workers=0, drop_last=True)

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

            while True:
                start_loop = time.time()
                frame_data = device.get_next_frame()

                data_all_antennas = []
                for i_ant in range(num_rx_antennas):
                    mat = frame_data[i_ant, :, :]
                    dfft_dbfs = algo.compute_doppler_map(mat, i_ant)
                    data_all_antennas.append(dfft_dbfs)

                range_doppler = do_inference_processing(data_all_antennas)
                self.debouncer.add_scan(range_doppler)

                dtm, rtm = self.debouncer.get_scans()
                rtm_tensor = torch.stack(rtm, dim=1).float().squeeze(2)
                dtm_tensor = torch.stack(dtm, dim=1).float().squeeze(2)

                # rdtm = torch.stack([rtm_tensor, dtm_tensor], dim=1).unsqueeze(0)
                rdtm = torch.stack([rtm_tensor, dtm_tensor], dim=1).unsqueeze(0)
                rdtm = rdtm.permute(0, 2, 1, 3)
                print(f"shape of input rdtm: {rdtm.shape}")
                
                prediction = np.zeros((1, self.num_classes))
                if rdtm.shape[3] >= self.observation_length:
                    output = self.model(rdtm)  
                    print(f"shape of output: {output.shape}")
                    output = torch.softmax(output, dim=1)  
                    prediction = output.squeeze(0).cpu().detach().numpy()

                    # max index 
                    max_idx = torch.argmax(output, dim=1).item()
                    label = self.dataset.get_class_name(max_idx)
                    print(f"output shape: {output.shape}, max index: {max_idx}, label: {label}")
                    print(f"[RTM] Detected class: {label} with probability: {prediction.max() * 100:.2f}%")


                if self.prev_rtm_tensor is not None:
                    if rtm_tensor.shape == self.prev_rtm_tensor.shape:
                        diff = (rtm_tensor - self.prev_rtm_tensor).abs()
                        max_diff = diff.max().item()
                        mean_diff = diff.mean().item()
                        # print(f"[RTM] max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")
                    else:
                        print(f"[RTM] Shape mismatch: {rtm_tensor.shape} vs {self.prev_rtm_tensor.shape}")
                else:
                    print("[RTM] First frame received.")

                self.prev_rtm_tensor = rtm_tensor.clone()

                self.visualizer.add_prediction_step(dtm_tensor.numpy(), rtm_tensor.numpy(), prediction)

                end_loop = time.time()
                elapsed_time = end_loop-start_loop
                sleep_time = max(0,0.1-elapsed_time)
                print(f"[RTM] Loop time: {elapsed_time:.4f}s, sleeping for: {sleep_time:.4f}s")
                time.sleep(sleep_time)

    def start_gui(self):
        self.visualizer.start_gui()


class Visualizer:
    def __init__(self, observation_length, num_classes):
        self.fig = plt.figure(figsize=(18, 10))
        self.prob_history_length = observation_length
        self.num_classes = num_classes

        self.rtm_buffer = np.zeros((32, self.prob_history_length))
        self.dtm_buffer = np.zeros((32, self.prob_history_length))



        self.vmin = 0.0
        self.vmax = 0.8

        self.plots = list(self.prepare_range_doppler_subplots())
        self.anim = FuncAnimation(self.fig, self.visualize_data, interval=100)

        self.pred_history = pd.DataFrame(columns=np.arange(0, self.num_classes, 1))
        # self.fig2 = plt.figure(figsize=(18, 10))
        # self.plots2 = list(self.prepare_prediction_subplots())
        # self.anim2 = FuncAnimation(self.fig2, self.visualize_pred, interval=100)

    def start_gui(self):
        print("Starting GUI...")
        plt.show()

    def prepare_range_doppler_subplots(self):
        gs = GridSpec(nrows=2, ncols=1, hspace=0.5)

        ax_range = self.fig.add_subplot(gs[0, 0])
        ax_range.set_title('Range over Time')
        ax_range.set_xlabel('Time (frames)')
        ax_range.set_ylabel('Range (m)')
        img1 = ax_range.imshow(np.zeros((32, self.prob_history_length)), aspect='auto', origin='lower', vmin=self.vmin, vmax=self.vmax)
        self.fig.colorbar(img1, ax=ax_range, orientation='vertical', label='Intensity')
        yield img1

        ax_velocity = self.fig.add_subplot(gs[1, 0])
        ax_velocity.set_title('Velocity over Time')
        ax_velocity.set_xlabel('Time (frames)')
        ax_velocity.set_ylabel('Velocity (m/s)')
        # ax_velocity.set_ylim(-1, 1)
        img2 = ax_velocity.imshow(np.zeros((32, self.prob_history_length)), aspect='auto', origin='lower', vmin=0.0, vmax=1.25)
        self.fig.colorbar(img2, ax=ax_velocity, orientation='vertical', label='Intensity')
        yield img2

    def prepare_prediction_subplots(self):
        gs = GridSpec(nrows=6, ncols=2, hspace=0.5)

        for idx in range(self.num_classes):
            ax = self.fig2.add_subplot(gs[idx // 2, idx % 2])
            ax.set_title(f'Class {idx}')
            ax.set_xlabel('Time (frames)')
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            line, = ax.plot([], [], label=f'Class {idx}')
            self.pred_history[idx] = pd.Series(dtype=float)
            yield line

    def visualize_pred(self, _):
        for index, (_, column_data) in enumerate(self.pred_history.items()):
            if len(column_data) > 0:
                self.plots2[index].set_data(range(len(column_data)), column_data.values)

    def add_prediction_step(self, dtm, rtm, prediction):
        if len(self.pred_history) >= self.prob_history_length:
            self.pred_history = self.pred_history.iloc[1:]

        pred_series = pd.Series(prediction.flatten(), index=self.pred_history.columns)
        self.pred_history = pd.concat([self.pred_history, pd.DataFrame([pred_series])], ignore_index=True)

        if rtm.shape[1] != 100:
            self.rtm_buffer = np.roll(self.rtm_buffer, -1, axis=1)
            self.dtm_buffer = np.roll(self.dtm_buffer, -1, axis=1)
            self.rtm_buffer[:, -1] = rtm[:, -1]
            self.dtm_buffer[:, -1] = dtm[:, -1]
        else:
            self.rtm_buffer = rtm
            self.dtm_buffer = dtm

    def visualize_data(self, frame):
        if self.rtm_buffer is None or self.dtm_buffer is None:
            return
        
        scale_v = 2
        self.plots[0].set_data(self.rtm_buffer)
        self.plots[1].set_data(self.dtm_buffer)


if __name__ == "__main__":
    observation_length = 10
    num_classes = 4

    inference = PredictionInference(observation_length=observation_length, num_classes=num_classes)
    stop_event = threading.Event()


    def run_inference():
        try:
            inference.run()
        except Exception as e:
            print(f"[Thread] Error: {e}")
            traceback.print_exc() 


    t = threading.Thread(target=run_inference)
    t.daemon = True
    t.start()

    try:
        inference.start_gui()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received. Exiting...")
        stop_event.set()
        plt.close('all')
        t.join(timeout=1)
        print("[Main] Shutdown complete.")