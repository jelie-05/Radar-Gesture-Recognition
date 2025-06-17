from src.internal.fft_spectrum import *
from src.AvianRDKWrapper.ifxRadarSDK import *
from src.utils.doppler import DopplerAlgo
from src.utils.common import do_inference_processing, do_preprocessing
from src.utils.debouncer_time import DebouncerTime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import pandas as pd
import tkinter as tk

class AnnotationLoad:
    def __init__(self, observation_length, num_classes):
        self.num_classes = num_classes
        self.observation_length = observation_length

        self.debouncer = DebouncerTime(memory_length=self.observation_length,)
        self.prev_rtm_tensor = None
        
        self.gesture_label = 'nothing'
        self.id = 13

        self.recording_path = f'data/recording/{self.gesture_label}/record_{self.gesture_label}_{self.id:04d}.npy'
        
        self.loaded_recording = np.load(self.recording_path)  # shape: [frames, num_ant, height, width]
        self.loaded_frame_idx = 0

        self.global_marked_frames = []
        self.recording_name = os.path.basename(self.recording_path)
        

    def load_next_frame(self):
        if self.loaded_frame_idx >= len(self.loaded_recording):
            print("[Loader] End of recording reached.")
            return None  # or raise StopIteration

        frame = self.loaded_recording[self.loaded_frame_idx]
        self.loaded_frame_idx += 1
        return frame

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

            rtm_chunk, dtm_chunk = [], []
            chunk_size = 100
            chunk_idx = 0

            while True:
                frame_data = self.load_next_frame()
                if frame_data is None:
                    print("[Run] Finished all frames.")
                    break

                data_all_antennas = []
                for i_ant in range(num_rx_antennas):
                    mat = frame_data[i_ant, :, :]
                    dfft_dbfs = algo.compute_doppler_map(mat, i_ant)
                    data_all_antennas.append(dfft_dbfs)

                range_doppler = do_inference_processing(data_all_antennas)
                self.debouncer.add_scan(range_doppler)

                dtm, rtm = self.debouncer.get_scans()
                if len(rtm) > 0 and len(dtm) > 0:
                    rtm_chunk.append(rtm[-1])
                    dtm_chunk.append(dtm[-1])

                if len(rtm_chunk) == chunk_size:
                    self.plot_chunk(rtm_chunk, dtm_chunk, chunk_idx)
                    rtm_chunk, dtm_chunk = [], []
                    chunk_idx += 1

            # Plot remaining frames
            if len(rtm_chunk) > 0:
                self.plot_chunk(rtm_chunk, dtm_chunk, chunk_idx)

            annotation = {
                'file_name': self.recording_name,
                'gesture': self.gesture_label,
                'start_frames': str(self.global_marked_frames),
                'num_starts': len(self.global_marked_frames)
            }
            df = pd.DataFrame([annotation])

            csv_path = "data/recording/annotation.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True) 

            if os.path.exists(csv_path):
                existing = pd.read_csv(csv_path)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_csv(csv_path, index=False)
            print(f"Saved annotations to {csv_path}")

    def move_window_to_left(fig):
        try:
            manager = plt.get_current_fig_manager()
            root = manager.window  # This is the actual Tkinter window
            root.update_idletasks()  # Ensure geometry is ready
            root.geometry("+0+50")   # Move to left side (0px from left, 50px from top)
        except Exception as e:
            print(f"⚠️ Could not move window: {e}")

    def plot_chunk(self, rtm_list, dtm_list, chunk_idx):
        rtm_array = np.stack(rtm_list)
        dtm_array = np.stack(dtm_list)
        chunk_start_frame = chunk_idx * 100
        chunk_end_frame = chunk_start_frame + rtm_array.shape[0]
        self.marked_frames = []
        self.skip_chunk = False  # Reset skip flag

        rtm_plot = np.mean(rtm_array, axis=2).T
        dtm_plot = np.mean(dtm_array, axis=2).T

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))
        extent_rtm = [chunk_start_frame, chunk_end_frame, 0, rtm_plot.shape[0]]
        extent_dtm = [chunk_start_frame, chunk_end_frame, 0, dtm_plot.shape[0]]

        im1 = ax1.imshow(rtm_plot, aspect='auto', origin='lower', cmap='jet', extent=extent_rtm)
        im2 = ax2.imshow(dtm_plot, aspect='auto', origin='lower', cmap='jet', extent=extent_dtm)

        ax1.set_title(f'RTM: Chunk {chunk_idx} [{extent_rtm[0]}–{extent_rtm[1]-1}]')
        ax2.set_title(f'DTM: Chunk {chunk_idx} [{extent_dtm[0]}–{extent_dtm[1]-1}]')

        xticks = np.arange(chunk_start_frame, chunk_end_frame, 5)
        ax1.set_xticks(xticks)
        ax2.set_xticks(xticks)

        ax1.set_xlabel('Frame index (global)')
        ax1.set_ylabel('Range bin')
        ax2.set_xlabel('Frame index (global)')
        ax2.set_ylabel('Velocity bin')

        fig.colorbar(im1, ax=ax1)
        fig.colorbar(im2, ax=ax2)
        plt.tight_layout()

        # Move window to left
        try:
            manager = plt.get_current_fig_manager()
            root = manager.window
            root.update_idletasks()

            def move_window_later():
                try:
                    root.geometry("+0+50")
                except Exception as e:
                    print(f"⚠️ Could not move window: {e}")

            root.after(100, move_window_later)  # ⏱ Schedule delayed move
        except Exception as e:
            print(f"⚠️ Could not schedule window move: {e}")

        # Define key press handler
        def on_key(event):
            if event.key == 'q':
                self.skip_chunk = True
                print("🚪 Skipping chunk...")
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=False)

        # input("🔎 Review plot. Press Enter to annotate or press 'q' in plot to skip...")

        # Only prompt for annotation if not skipped
        if not self.skip_chunk:
            # frame_input = input(f"🖋️ Enter global frame indices for Chunk {chunk_idx} (comma-separated), or press Enter to skip: ")
            while True:
                frame_input = input(f"🖋️ Enter global frame indices for Chunk {chunk_idx} (comma-separated), or type 'q' to skip to next: ").strip()

                if frame_input.lower() == 'q':
                    print("🚪 Skipping to next chunk.")
                    break  # move to next chunk

                if frame_input == '':
                    print("ℹ️  No input given. Still in annotation mode.")
                    continue  # stay in loop

                try:
                    entries = [int(idx.strip()) for idx in frame_input.split(',') if idx.strip()]
                    for abs_idx in entries:
                        if chunk_start_frame <= abs_idx < chunk_end_frame:
                            self.marked_frames.append(abs_idx)
                            print(f"✅ Marked global frame {abs_idx}")
                        else:
                            print(f"⚠️ Frame {abs_idx} out of range ({chunk_start_frame}–{chunk_end_frame - 1})")
                except ValueError:
                    print("⚠️ Invalid input, must be comma-separated integers or 'q'")

        plt.close(fig)
        self.global_marked_frames.extend(self.marked_frames)



if __name__ == "__main__":
    observation_length = 10
    num_classes = 4

    inference = AnnotationLoad(observation_length=observation_length, num_classes=num_classes)
    inference.run()