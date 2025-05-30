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


class PredictionInference:
    def __init__(self):
        self.debouncer = DebouncerTime()
        self.visualizer = Visualizer()
        self.prev_rtm_tensor = None  # For checking change

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

                # Check if RTM changes
                if self.prev_rtm_tensor is not None:
                    if rtm_tensor.shape == self.prev_rtm_tensor.shape:
                        diff = (rtm_tensor - self.prev_rtm_tensor).abs()
                        max_diff = diff.max().item()
                        mean_diff = diff.mean().item()
                        print(f"[RTM] max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f}")
                    else:
                        print(f"[RTM] Shape mismatch: {rtm_tensor.shape} vs {self.prev_rtm_tensor.shape}")
                else:
                    print("[RTM] First frame received.")

                self.prev_rtm_tensor = rtm_tensor.clone()

                # Send data to visualizer
                self.visualizer.add_prediction_step(dtm_tensor.numpy(), rtm_tensor.numpy())

    def start_gui(self):
        self.visualizer.start_gui()


class Visualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(18, 10))
        self.prob_history_length = 100

        # Rolling buffers
        self.rtm_buffer = np.zeros((32, self.prob_history_length))
        self.dtm_buffer = np.zeros((32, self.prob_history_length))

        # Manual color range
        self.vmin = 0.0
        self.vmax = 0.8

        self.plots = list(self.prepare_range_doppler_subplots())
        self.anim = FuncAnimation(self.fig, self.visualize_data, interval=100)

    def start_gui(self):
        print("Starting GUI...")
        plt.show()

    def prepare_range_doppler_subplots(self):
        gs = GridSpec(nrows=2, ncols=1, hspace=0.5)

        # --- Range-Time Plot ---
        ax_range = self.fig.add_subplot(gs[0, 0])
        ax_range.set_title('Range over Time')
        ax_range.set_xlabel('Time (frames)')
        ax_range.set_ylabel('Range (m)')
        img1 = ax_range.imshow(
            np.zeros((32, 100)), aspect='auto', origin='lower',
            vmin=self.vmin, vmax=self.vmax
        )
        self.fig.colorbar(img1, ax=ax_range, orientation='vertical', label='Intensity')
        yield img1

        # --- Doppler-Time Plot ---
        ax_velocity = self.fig.add_subplot(gs[1, 0])
        ax_velocity.set_title('Velocity over Time')
        ax_velocity.set_xlabel('Time (frames)')
        ax_velocity.set_ylabel('Velocity (m/s)')
        ax_velocity.set_ylim(-3, 3)
        img2 = ax_velocity.imshow(
            np.zeros((32, 100)), aspect='auto', origin='lower',
            vmin=self.vmin, vmax=self.vmax
        )
        self.fig.colorbar(img2, ax=ax_velocity, orientation='vertical', label='Intensity')
        yield img2

    def add_prediction_step(self, dtm, rtm):
        if rtm.shape[1] != 100:
            # shift buffer and append last column
            self.rtm_buffer = np.roll(self.rtm_buffer, -1, axis=1)
            self.dtm_buffer = np.roll(self.dtm_buffer, -1, axis=1)

            self.rtm_buffer[:, -1] = rtm[:, -1]
            self.dtm_buffer[:, -1] = dtm[:, -1]
        else:
            # Replace whole buffer (for debugging/initialization)
            self.rtm_buffer = rtm
            self.dtm_buffer = dtm

    def visualize_data(self, frame):
        if self.rtm_buffer is None or self.dtm_buffer is None:
            return

        self.plots[0].set_data(self.rtm_buffer)
        self.plots[1].set_data(self.dtm_buffer)


if __name__ == "__main__":
    inference = PredictionInference()
    stop_event = threading.Event()

    def run_inference():
        try:
            inference.run()
        except Exception as e:
            print(f"[Thread] Error: {e}")

    t = threading.Thread(target=run_inference)
    t.daemon = True
    t.start()

    try:
        inference.start_gui()  # runs until you close the plot
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received. Exiting...")
        stop_event.set()
        t.join(timeout=1)
        print("[Main] Shutdown complete.")

