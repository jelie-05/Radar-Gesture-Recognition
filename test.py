from ifxAvian import Avian
from src.utils.doppler_avian import DopplerAlgo

config = Avian.DeviceConfig(
    sample_rate_Hz = 2500000,       # 1MHZ
    rx_mask = 7,                      # activate RX1 and RX3
    tx_mask = 1,                      # activate TX1
    if_gain_dB = 25,                  # gain of 33dB
    tx_power_level = 31,              # TX power level of 31
    start_frequency_Hz = 57569828864,        # 60GHz 
    end_frequency_Hz = 63930171392,        # 61.5GHz
    num_chirps_per_frame = 256,       # 128 chirps per frame
    num_samples_per_chirp = 128,       # 64 samples per chirp
    chirp_repetition_time_s = 0.0004112379392609, # 0.5ms
    frame_repetition_time_s = 0.10526315867900848,   # 0.15s, frame_Rate = 6.667Hz
    mimo_mode = 'off'                 # MIMO disabled
)
num_rx_antennas = 3
algo = DopplerAlgo(config, num_rx_antennas)
