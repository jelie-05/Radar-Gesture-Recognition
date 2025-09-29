from src.internal.fft_spectrum import *
from src.AvianRDKWrapper.ifxRadarSDK import *
from src.utils.doppler_avian import DopplerAlgo
from src.utils.common import do_inference_processing, do_inference_processing_RAM
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
from src.train_utils.dataset import RadarGestureDataset
import time
from src.utils.DBF import DBF
import queue
import json
from ifxAvian import Avian


class LivePlot:
    def __init__(self, max_angle_degrees: float, max_range_m: float, data_queue: queue.Queue):
        self.h = None
        self.max_angle_degrees = max_angle_degrees
        self.max_range_m = max_range_m
        self.queue = data_queue

        self._fig, self._ax = plt.subplots(nrows=1, ncols=1)
        self._fig.canvas.manager.set_window_title("Range-Angle-Map using DBF")
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.0, 0.03, 1])
        self.colorbar_axis = cbar_ax

        # Add animation callback
        self.anim = FuncAnimation(self._fig, self._update_plot, interval=100)

    def _draw_first_time(self, data: np.ndarray):
        self.h = self._ax.imshow(
            data,
            vmin=-60,
            vmax=0,
            cmap='viridis',
            extent=(-self.max_angle_degrees,
                    self.max_angle_degrees,
                    0,
                    self.max_range_m),
            origin='lower')
        self._ax.set_xlabel("angle (degrees)")
        self._ax.set_ylabel("distance (m)")
        self._ax.set_aspect("auto")

        self._cbar = self._fig.colorbar(self.h, cax=self.colorbar_axis)
        self._cbar.ax.set_ylabel("magnitude (a.u.)")

    def _draw_next_time(self, data: np.ndarray):
        self.h.set_data(data)

    def _update_plot(self, frame):
        if not self.queue.empty():
            data, title = self.queue.get_nowait()
            if self.h is None:
                self._draw_first_time(data)
            else:
                self._draw_next_time(data)
            self._ax.set_title(title)
            self._fig.canvas.draw_idle()

    def close(self, event=None):
        if not self.is_closed():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')

    def is_closed(self):
        return not self._is_window_open


class PredictionInference:
    def __init__(self, observation_length, num_classes):
        self.num_classes = num_classes
        self.observation_length = observation_length

        # Initialize the device       
        radar_config, config = self.get_config()
        Avian.Device().set_config(config)

        self.num_rx_antennas = radar_config["num_rx_antennas"]
        self.num_beams = radar_config["num_beams"]
        self.max_angle_degrees = radar_config["max_angle_degrees"]
        self.max_range_m = 1.2

        input_channel = 3

        # Initialize Visualizer
        self.debouncer = DebouncerTime(memory_length=self.observation_length,)
        self.visualizer = Visualizer(observation_length=self.observation_length, num_classes=num_classes)
        self.prev_rtm_tensor = None
        self.ra_queue = queue.Queue()
        self.plot = LivePlot(self.max_angle_degrees, self.max_range_m, self.ra_queue)

        # Initialize the model
        self.model = SimpleCNN(in_channels=input_channel, num_classes=self.num_classes)
        self.model.eval()
        run_id = 'run_250805_02'
        output_path = f'outputs/radargesture/{run_id}/'
        model_path = os.path.join(output_path, 'checkpoints/best_model.pth')

        mapping_path = '/home/swadiryus/projects/dataset_transformed/labels_mapping.json'
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)
            print(f"mapping: {self.mapping}")

        if os.path.exists(model_path):
            # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])

            print("[Model] Loaded successfully.")
            input("[Model] Press Enter to continue...")
        else:
            input("[Model] Model file not found. Please ensure the path is correct and the model is trained.")
        self.model = self.model.to('cpu')

    def run(self):
        with Avian.Device() as device:
            dev_config = device.get_config()
            algo = DopplerAlgo(dev_config, self.num_rx_antennas)
            dbf = DBF(self.num_rx_antennas, num_beams = self.num_beams, max_angle_degrees =self.max_angle_degrees)

            while True:
                start_loop = time.time()
                frame_data = device.get_next_frame()

                data_all_antennas = []
                rd_spectrum = np.zeros((dev_config.num_samples_per_chirp, 2*dev_config.num_chirps_per_frame, self.num_rx_antennas), dtype=complex)
                for i_ant in range(self.num_rx_antennas):
                    dfft_dbfs = algo.compute_doppler_map(frame_data[i_ant], i_ant)
                    data_all_antennas.append(dfft_dbfs)
                    rd_spectrum[:,:,i_ant] = dfft_dbfs

                ####################### ANGLE MAP PROCESSING #######################
                # # Rearrange data for DBF
                # data_all_antennas_np = np.stack(data_all_antennas, axis=0)
                # data_all_antennas_np = data_all_antennas_np.transpose(1, 2, 0)

                # num_samples_per_chirp = data_all_antennas_np.shape[0]

                # rd_beam_formed = dbf.run(data_all_antennas_np)
                rd_beam_formed = dbf.run(rd_spectrum)

                beam_range_energy = np.zeros((dev_config.num_samples_per_chirp, self.num_beams))
                for i_beam in range(self.num_beams):
                    doppler_i = rd_beam_formed[:,:,i_beam]
                    beam_range_energy[:,i_beam] += np.linalg.norm(doppler_i, axis=1) / np.sqrt(self.num_beams)

                max_energy = beam_range_energy.max()
                scale = 150
                beam_range_energy = scale * (beam_range_energy / max_energy - 1)

                # Find dominant angle of target for visualization
                _, idx = np.unravel_index(beam_range_energy.argmax(), beam_range_energy.shape)
                angle_degrees = np.linspace(-self.max_angle_degrees, self.max_angle_degrees, self.num_beams)[idx]

                # Plot Range-Angle Map
                # self.plot.draw(beam_range_energy, f"Range-Angle map using DBF, angle={angle_degrees:+02.0f} degrees")
                self.ra_queue.put((beam_range_energy.copy(), f"Range-Angle map using DBF, angle={angle_degrees:+02.0f} degrees"))

                ################# RANGE-DOPPLER MAP PROCESSING #################
                # Range-Doppler Map
                range_angle = do_inference_processing_RAM(beam_range_energy)
                range_doppler = do_inference_processing(data_all_antennas)
                self.debouncer.add_scan(range_doppler, ram=range_angle)

                dtm, rtm, atm = self.debouncer.get_scans()   # Only the first channel is used

                rtm_tensor = torch.stack(rtm, dim=1).float().squeeze(2)
                dtm_tensor = torch.stack(dtm, dim=1).float().squeeze(2)
                atm_tensor = torch.stack(atm, dim=1).float().squeeze(2) if atm else None

                rdatm = torch.stack([rtm_tensor, dtm_tensor, atm_tensor], dim=1).unsqueeze(0)    # (1, N, 2 or 3, T)
                rdatm = rdatm.permute(0, 2, 1, 3)
                
                prediction = np.zeros((1, self.num_classes))
                if rdatm.shape[3] >= self.observation_length:
                    output = self.model(rdatm[:,:,:,:])  
                    output = torch.softmax(output, dim=1)  
                    prediction = output.squeeze(0).cpu().detach().numpy()

                    # max index 
                    max_idx = torch.argmax(output, dim=1).item()
                    label = self.get_class_name(max_idx)
                    print(f"[RTM] Detected class: {label}; idx: {max_idx} with probability: {prediction.max() * 100:.2f}%")


                if self.prev_rtm_tensor is not None:
                    if rtm_tensor.shape == self.prev_rtm_tensor.shape:
                        diff = (rtm_tensor - self.prev_rtm_tensor).abs()
                        # max_diff = diff.max().item()
                        # mean_diff = diff.mean().item()
                        # # print(f"[RTM] max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")
                    else:
                        print(f"[RTM] Shape mismatch: {rtm_tensor.shape} vs {self.prev_rtm_tensor.shape}")
                else:
                    print("[RTM] First frame received.")

                self.prev_rtm_tensor = rtm_tensor.clone()
                # TODO: Add ATM to visualizer
                self.visualizer.add_prediction_step(dtm_tensor.numpy(), rtm_tensor.numpy(), atm_tensor.numpy(), prediction)

                end_loop = time.time()
                elapsed_time = end_loop-start_loop
                sleep_time = max(0,0.03-elapsed_time)
                print(f"[RTM] Loop time: {elapsed_time:.4f}s, sleeping for: {sleep_time:.4f}s")
                time.sleep(sleep_time)

    def start_gui(self):
        self.visualizer.start_gui()

    def get_class_name(self, idx):
        return self.mapping.get(str(idx), f"Unknown class {idx}")

    def get_config(self):
        dev_config = {
            'sample_rate_Hz': 2000000,
            'rx_mask': 7,
            'tx_mask': 1,
            'if_gain_dB': 31,
            'tx_power_level': 31,
            'start_frequency_Hz': 58.5e9,
            'end_frequency_Hz': 62.5e9,
            'num_chirps_per_frame': 32,
            'num_samples_per_chirp': 64,
            'chirp_repetition_time_s': 0.0003,
            'frame_repetition_time_s': 0.03,
            'mimo_mode': 'off'
        }  
        radar_config = {'dev_config': dev_config, 
                        'num_rx_antennas': 3, 
                        'num_beams': 32,
                        'max_angle_degrees': 55}

        avian_config = Avian.DeviceConfig(
            sample_rate_Hz = dev_config['sample_rate_Hz'],
            rx_mask = dev_config['rx_mask'],
            tx_mask = dev_config['tx_mask'],
            if_gain_dB = dev_config['if_gain_dB'],
            tx_power_level = dev_config['tx_power_level'],
            start_frequency_Hz = dev_config['start_frequency_Hz'],
            end_frequency_Hz = dev_config['end_frequency_Hz'],
            num_chirps_per_frame = dev_config['num_chirps_per_frame'],
            num_samples_per_chirp = dev_config['num_samples_per_chirp'],
            chirp_repetition_time_s = dev_config['chirp_repetition_time_s'],
            frame_repetition_time_s = dev_config['frame_repetition_time_s'],
            mimo_mode = dev_config['mimo_mode']
        )

        return radar_config, avian_config


class Visualizer:
    def __init__(self, observation_length, num_classes):
        self.fig = plt.figure(figsize=(18, 10))
        self.prob_history_length = observation_length
        self.num_classes = num_classes

        self.rtm_buffer = np.zeros((32, self.prob_history_length))
        self.dtm_buffer = np.zeros((32, self.prob_history_length))
        self.atm_buffer = np.zeros((32, self.prob_history_length))

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
        gs = GridSpec(nrows=3, ncols=1, hspace=0.5)

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

        ax_angle = self.fig.add_subplot(gs[2, 0])
        ax_angle.set_title('Angle over Time')
        ax_angle.set_xlabel('Time (frames)')
        ax_angle.set_ylabel('Angle (°)')
        img3 = ax_angle.imshow(np.zeros((32, self.prob_history_length)), aspect='auto', origin='lower', vmin=-100, vmax=0)
        self.fig.colorbar(img3, ax=ax_angle, orientation='vertical', label='Intensity')
        yield img3

    def prepare_prediction_subplots(self):
        gs = GridSpec(nrows=3, ncols=1, hspace=0.5)

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

    def add_prediction_step(self, dtm, rtm, atm, prediction):

        if len(self.pred_history) >= self.prob_history_length:
            self.pred_history = self.pred_history.iloc[1:]

        pred_series = pd.Series(prediction.flatten(), index=self.pred_history.columns)
        self.pred_history = pd.concat([self.pred_history, pd.DataFrame([pred_series])], ignore_index=True)

        if rtm.shape[1] != 100:
            self.rtm_buffer = np.roll(self.rtm_buffer, -1, axis=1)
            self.dtm_buffer = np.roll(self.dtm_buffer, -1, axis=1)
            self.atm_buffer = np.roll(self.atm_buffer, -1, axis=1)

            self.rtm_buffer[:, -1] = rtm[:, -1]
            self.dtm_buffer[:, -1] = dtm[:, -1]
            self.atm_buffer[:, -1] = atm[:, -1]
        else:
            self.rtm_buffer = rtm
            self.dtm_buffer = dtm
            self.atm_buffer = atm

    def visualize_data(self, frame):
        if self.rtm_buffer is None or self.dtm_buffer is None:
            return
        
        scale_v = 2
        self.plots[0].set_data(self.rtm_buffer)
        self.plots[1].set_data(self.dtm_buffer)
        self.plots[2].set_data(self.atm_buffer)


if __name__ == "__main__":
    observation_length = 30
    num_classes = 6

    inference = PredictionInference(observation_length=observation_length, num_classes=num_classes)
    stop_event = threading.Event()

    def run_model_only():
        try:
            inference.run()
        except Exception as e:
            print(f"[Thread] Error: {e}")
            traceback.print_exc()

    # Start model-only part in background thread
    t = threading.Thread(target=run_model_only)
    t.daemon = True
    t.start()

    try:
        # GUI-related functions (including plt.show()) must run in the main thread
        inference.start_gui()
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received. Exiting...")
        
        stop_event.set()
        plt.close('all')
        t.join(timeout=1)
        print("[Main] Shutdown complete.")
