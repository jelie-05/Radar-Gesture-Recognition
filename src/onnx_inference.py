from src.internal.fft_spectrum import *
from src.AvianRDKWrapper.ifxRadarSDK import *
from src.utils.doppler import DopplerAlgo
from src.utils.common import do_inference_processing, do_preprocessing
from src.utils.debouncer_time import DebouncerTime
import threading
import numpy as np
import pandas as pd
import pickle
import time
import tflite_runtime.interpreter as tflite

def softmax_np(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)  # for numerical stability
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class PredictionInference:
    def __init__(self, observation_length, num_classes):
        self.num_classes = num_classes
        self.observation_length = observation_length

        self.debouncer = DebouncerTime(memory_length=self.observation_length,)
        # self.ort_session = ort.InferenceSession("runs/trained_models/train_0606.onnx")


        self.interpreter = tflite.Interpreter(model_path="runs/trained_models/my_custom_model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        with open('runs/trained_models/train_0606-idx_mapping.pkl', 'rb') as f:
            self.idx_to_class = pickle.load(f)
        

    def get_class_name(self, label):
        return self.idx_to_class[label]  

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

                rtm_np = np.stack(rtm, axis=1) 
                dtm_np = np.stack(dtm, axis=1)
                rtm_np = np.squeeze(rtm_np, axis=2)
                dtm_np = np.squeeze(dtm_np, axis=2)
                rdtm_np = np.stack([rtm_np, dtm_np], axis=1) 
                rdtm_np = np.expand_dims(rdtm_np, axis=0)    
                rdtm_np = rdtm_np.transpose(0, 2, 1, 3)

                
                if rdtm_np.shape[3] >= self.observation_length:
                    # output =  self.ort_session.run(None, {"input": rdtm_np}) 
                    # output = np.array(output)
                    # output = np.squeeze(output, axis=0)  
                    self.interpreter.set_tensor(self.input_details[0]['index'], rdtm_np)
                    self.interpreter.invoke()
                    output = self.interpreter.get_tensor(self.output_details[0]['index'])
                    output = np.array(output)
                    if output.ndim == 1:
                        output = output[None, :]  
                    print(f"[DEBUG] output shape BEFORE softmax: {output.shape}")
                    
                    # Apply softmax to the output manually using NumPy
                    exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))  
                    softmax_output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
                    
                    # Get the prediction
                    prediction = softmax_output.squeeze(0) 
                    max_idx = np.argmax(softmax_output, axis=1).item()  
                    
                    # Get the class name from the index
                    label = self.get_class_name(max_idx)
                    print(f"output shape: {output.shape}, max index: {max_idx}, label: {label}")
                    print(f"[RTM] Detected class: {label} with probability: {prediction.max() * 100:.2f}%")


                end_loop = time.time()
                elapsed_time = end_loop-start_loop
                sleep_time = max(0,0.1-elapsed_time)
                print(f"[RTM] Loop time: {elapsed_time:.4f}s, sleeping for: {sleep_time:.4f}s")
                time.sleep(sleep_time)


if __name__ == "__main__":
    observation_length = 10
    num_classes = 4

    inference = PredictionInference(observation_length=observation_length, num_classes=num_classes)
    stop_event = threading.Event()

    inference.run()