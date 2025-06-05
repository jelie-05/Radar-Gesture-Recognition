# ===========================================================================
# Copyright (C) 2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from ifxAvian import Avian
from internal.fft_spectrum import *

class DopplerAlgo:
    """Compute Range-Doppler map"""

    def __init__(self, config : Avian.DeviceConfig, num_ant : int, mti_alpha : float = 0.8):
        """Create Range-Doppler map object

        Parameters:
            - config:    Radar configuration as returned by get_config() method
            - num_ant:   Number of antennas
            - mti_alpha: Parameter alpha of Moving Target Indicator
        """
        self.num_chirps_per_frame = config.num_chirps_per_frame
        num_samples_per_chirp = config.num_samples_per_chirp

        # compute Blackman-Harris Window matrix over chirp samples(range)
        self.range_window = signal.windows.blackmanharris(num_samples_per_chirp).reshape(1,num_samples_per_chirp)

        # compute Blackman-Harris Window matrix over number of chirps(velocity)
        self.doppler_window = signal.windows.blackmanharris(self.num_chirps_per_frame).reshape(1,self.num_chirps_per_frame)

        # parameter for moving target indicator (MTI)
        self.mti_alpha = mti_alpha

        # initialize MTI filter
        self.mti_history = np.zeros((self.num_chirps_per_frame, num_samples_per_chirp, num_ant))

    def compute_doppler_map(self, data : np.ndarray, i_ant : int):
        """Compute Range-Doppler map for i-th antennas

        Parameter:
            - data:     Raw-data for one antenna (dimension:
                        num_chirps_per-frame x num_samples_per_chirp)
            - i_ant:    Number of antenna
        """
        # Step 1 - Remove average from signal (mean removal)
        data = data - np.average(data)

        # Step 2 - MTI processing to remove static objects
        data_mti = data - self.mti_history[:,:,i_ant]
        self.mti_history[:,:,i_ant] = data*self.mti_alpha + self.mti_history[:,:,i_ant]*(1-self.mti_alpha)

        # Step 3 - calculate fft spectrum for the frame
        fft1d = fft_spectrum(data_mti, self.range_window)

        # prepare for doppler FFT

        # Transpose
        # Distance is now indicated on y axis
        fft1d = np.transpose(fft1d)

        # Step 4 - Windowing the Data in doppler
        fft1d = np.multiply(fft1d,self.doppler_window)

        zp2 = np.pad(fft1d,((0,0),(0,self.num_chirps_per_frame)), "constant")
        fft2d = np.fft.fft(zp2)/self.num_chirps_per_frame

        # re-arrange fft result for zero speed at centre
        return np.fft.fftshift(fft2d,(1,))

# -------------------------------------------------
# Presentation
# -------------------------------------------------
class Draw:
    # Represents drawing for example
    #
    # Draw is done for each antenna, and each antenna is represented for
    # other subplot

    def __init__(self, max_speed_m_s, max_range_m, num_ant):
        # max_range_m:   maximum supported range
        # max_speed_m_s: maximum supported speed
        # num_ant:       Number of antennas
        self._h = []
        self._max_speed_m_s = max_speed_m_s
        self._max_range_m = max_range_m
        self._num_ant = num_ant

        self._fig, ax = plt.subplots(nrows=1, ncols=num_ant, figsize=((num_ant+1)//2,2))
        if(num_ant==1):
            self._ax = [ax]
        else:
            self._ax = ax

        self._fig.canvas.manager.set_window_title("Doppler")
        self._fig.set_size_inches(3*num_ant+1, 3+1/num_ant)
        self._fig.canvas.mpl_connect('close_event', self.close)
        self._is_window_open = True

    def _draw_first_time(self, data_all_antennas):
        # First time draw
        #
        # It computes minimal, maximum value and draw data for all antennas
        # in same scale
        # data_all_antennas: array of raw data for each antenna

        minmin = min([np.min(data) for data in data_all_antennas])
        maxmax = max([np.max(data) for data in data_all_antennas])

        for i_ant in range(self._num_ant):
            data = data_all_antennas[i_ant]
            print(f"data size: {data.size}, data shape: {data.shape}")
            h = self._ax[i_ant].imshow(
                        data,
                        vmin=minmin, vmax=maxmax,
                        cmap='hot',
                        extent=(    -self._max_speed_m_s,
                                    self._max_speed_m_s,
                                    0,
                                    self._max_range_m),
                        origin='lower')
            self._h.append(h)

            self._ax[i_ant].set_xlabel("velocity (m/s)")
            self._ax[i_ant].set_ylabel("distance (m)")
            self._ax[i_ant].set_title("antenna #"+str(i_ant))
        self._fig.subplots_adjust(right=0.8)
        cbar_ax = self._fig.add_axes([0.85, 0.0, 0.03, 1])

        cbar =self._fig.colorbar(self._h[0], cax=cbar_ax)
        cbar.ax.set_ylabel("magnitude (dB)")

    def _draw_next_time(self, data_all_antennas):
        # Update data for each antenna

        for i_ant in range(0, self._num_ant):
            data = data_all_antennas[i_ant]
            self._h[i_ant].set_data(data)

    def draw(self, data_all_antennas):
        # Draw data for all antenna
        if self._is_window_open:
            first_run = len(self._h) == 0
            if first_run:
                self._draw_first_time(data_all_antennas)
            else:
                self._draw_next_time(data_all_antennas)

            # Needed for Matplotlib ver: 3.4.0 and 3.4.1 helps with capture closing event
            plt.draw()
            plt.pause(1e-3)

    def close(self, event = None):
        if self.is_open():
            self._is_window_open = False
            plt.close(self._fig)
            plt.close('all')
            print('Application closed!')

    def is_open(self):
        return self._is_window_open
# -------------------------------------------------
# Helpers
# -------------------------------------------------
def parse_attr_nframes_frate( description, def_nframes, def_frate):
    # Parse all program attributes
    # description:   describes program
    # def_nframes:   default number of frames
    # def_frate:     default frame rate in Hz
    parser = argparse.ArgumentParser(
        description=description)

    parser.add_argument('-n', '--nframes', type=int,
                        default=def_nframes, help="number of frames, default "+str(def_nframes))
    parser.add_argument('-f', '--frate', type=int, default=def_frate,
                        help="frame rate in Hz, default "+str(def_frate))

    return parser.parse_args()

def linear_to_dB(x):
    return 20*np.log10(abs(x))

# -------------------------------------------------
# Main logic
# -------------------------------------------------
if __name__ == '__main__':
    args = parse_attr_nframes_frate(
        '''Displays range doppler map''',
        def_nframes=50,
        def_frate=5)
    with Avian.Device() as device:
        # activate all available antennas
        num_rx_antennas = device.get_sensor_information()["num_rx_antennas"]
        rx_mask = (1 << num_rx_antennas) - 1

        # metric = Avian.DeviceMetrics(
        #     sample_rate_Hz =           1_000_000,
        #     range_resolution_m =       0.15,
        #     max_range_m =              4.8,
        #     max_speed_m_s =            2.45,
        #     speed_resolution_m_s =     0.2,
        #     frame_repetition_time_s =  1/args.frate,
        #     center_frequency_Hz =      60_750_000_000,
        #     rx_mask =                  rx_mask,
        #     tx_mask =                  1,
        #     tx_power_level =           31,
        #     if_gain_dB =               33
        # )

        metric = Avian.DeviceMetrics(
            sample_rate_Hz =           1_000_000,
            range_resolution_m =       0.04,
            max_range_m =              0.64,
            max_speed_m_s =            2.0,
            speed_resolution_m_s =     0.2,
            frame_repetition_time_s =  1/args.frate,
            center_frequency_Hz =      60_750_000_000,
            rx_mask =                  rx_mask,
            tx_mask =                  1,
            tx_power_level =           31,
            if_gain_dB =               33
        )

        config = device.metrics_to_config(metric)

        device.set_config(config)

        doppler = DopplerAlgo(config, num_rx_antennas)
        draw = Draw(
            metric.max_speed_m_s, \
            metric.max_range_m, \
            num_rx_antennas)

        # for frame_number in range(args.nframes): # For each frame
        # for frame_number in range(10000): # For each frame
        try:
            while True:
                if not draw.is_open():
                    break
                frame_data = device.get_next_frame()
                data_all_antennas = []
                for i_ant in range(0, num_rx_antennas): #For each antenna
                    mat = frame_data[i_ant, :, :]
                    dfft_dbfs = linear_to_dB(doppler.compute_doppler_map(mat, i_ant))
                    data_all_antennas.append(dfft_dbfs)
                draw.draw(data_all_antennas)

        except KeyboardInterrupt:
            print("end")

        draw.close()
