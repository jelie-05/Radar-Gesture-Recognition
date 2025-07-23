import torch
import torch.nn.functional as F

class DopplerAlgoTorch:
    """Compute batched Range-Doppler map in PyTorch using Blackman-Harris window"""

    def __init__(self, config: dict, num_ant: int, mti_alpha: float = 0.8, device='cpu'):
        self.num_chirps_per_frame = config["num_chirps_per_frame"]
        self.num_samples_per_chirp = config["num_samples_per_chirp"]
        self.num_ant = num_ant
        self.mti_alpha = mti_alpha
        self.device = device

        # Create true Blackman-Harris window
        self.range_window = self.blackmanharris_window(self.num_samples_per_chirp).view(1, 1, -1)  # (1, 1, N_samples)
        self.doppler_window = self.blackmanharris_window(self.num_chirps_per_frame).view(1, -1, 1)  # (1, N_chirps, 1)

        # MTI history buffer (stateful)
        self.mti_history = torch.zeros(
            num_ant, self.num_chirps_per_frame, self.num_samples_per_chirp,
            dtype=torch.float32, device=device
        )

    @staticmethod
    def blackmanharris_window(N, device='cpu'):
        """Manual Blackman-Harris window (4-term)"""
        n = torch.arange(N, dtype=torch.float32, device=device)
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        return (
            a0
            - a1 * torch.cos(2 * torch.pi * n / (N - 1))
            + a2 * torch.cos(4 * torch.pi * n / (N - 1))
            - a3 * torch.cos(6 * torch.pi * n / (N - 1))
        )

    def compute_doppler_map(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute Range-Doppler map for all antennas in batch.
        Input: data shape (N_antennas, N_chirps, N_samples)
        Output: doppler_map shape (N_antennas, N_samples, N_doppler_bins)
        """

        # Step 1: Remove average
        data = data - data.mean(dim=(1, 2), keepdim=True)

        # Step 2: MTI filtering
        data_mti = data - self.mti_history
        self.mti_history = self.mti_alpha * data + (1 - self.mti_alpha) * self.mti_history

        # Step 3: FFT over sample axis (range FFT)
        fft1d = self.fft_spectrum(data_mti)

        # Step 4: Doppler processing (FFT over chirps)
        fft1d = fft1d.transpose(1, 2)  # shape: (N_ant, N_samples, N_chirps)

        fft1d = fft1d * self.doppler_window.to(fft1d.device)

        # Zero pad and FFT along Doppler axis
        pad_amount = self.num_chirps_per_frame
        fft2d = F.pad(fft1d, (0, pad_amount), mode='constant', value=0)
        fft2d = torch.fft.fft(fft2d, dim=-1) / self.num_chirps_per_frame

        # FFT shift: center Doppler zero-frequency bin
        fft2d = torch.fft.fftshift(fft2d, dim=-1)

        return fft2d  # shape: (N_ant, N_samples, N_doppler_bins)

    def fft_spectrum(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply windowed FFT along sample axis with zero padding.
        Input: data shape (N_ant, N_chirps, N_samples)
        Output: fft1d shape (N_ant, N_chirps, N_range_bins)
        """

        # Remove DC bias per chirp
        data = data - data.mean(dim=2, keepdim=True)

        # Apply range window
        data = data * self.range_window.to(data.device)

        # Zero padding in sample axis
        data = F.pad(data, (0, self.num_samples_per_chirp), mode='constant', value=0)

        # FFT on padded sample axis
        fft = torch.fft.fft(data, dim=-1) / self.num_samples_per_chirp

        # Keep only positive frequencies
        fft = 2 * fft[..., :self.num_samples_per_chirp]

        return fft
