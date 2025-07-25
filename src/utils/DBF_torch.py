import torch
import math

class DBFTorch:
    def __init__(self, num_antennas: int, num_beams: int = 27,
                 max_angle_degrees: float = 45.0, d_by_lambda: float = 0.5,
                 device='cpu'):
        """
        Digital Beamforming constructor (PyTorch version)

        Args:
            num_antennas: number of RX antennas
            num_beams: number of output beams
            max_angle_degrees: field of view from -angle to +angle
            d_by_lambda: spacing between antennas in wavelength units
        """
        self.num_antennas = num_antennas
        self.num_beams = num_beams
        self.device = device

        angles = torch.linspace(
            -max_angle_degrees, max_angle_degrees, num_beams, device=device
        )
        angle_rad = angles * math.pi / 180.0  # degrees → radians

        # Shape: (num_antennas, num_beams)
        antenna_indices = torch.arange(num_antennas, device=device).view(-1, 1)
        phase_shifts = 2 * math.pi * d_by_lambda * antenna_indices * torch.sin(angle_rad)

        # Complex exponential weights, matching original reversed index order
        weights = torch.exp(1j * phase_shifts)
        weights = weights.flip(0)  # reverse antenna index as in original code

        self.weights = weights  # shape: (num_antennas, num_beams)

    def run(self, range_doppler: torch.Tensor) -> torch.Tensor:
        """
        Apply beamforming to range-doppler data

        Args:
            range_doppler: (num_samples, num_chirps, num_antennas) 

        Returns:
            beamformed_rdm: (num_samples, num_chirps, num_beams) 
        """
        assert range_doppler.shape[-1] == self.num_antennas
        if not torch.is_complex(range_doppler):
            raise ValueError("Input to DBF must be complex tensor")

        # Vectorized beamforming:
        # sum over antennas: output[n, c, b] = sum_a (X[n, c, a] * W[a, b])
        beamformed = torch.einsum('nca,ab->ncb', range_doppler, self.weights)
        return beamformed