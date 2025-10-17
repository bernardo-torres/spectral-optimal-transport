import torch
import torch.nn as nn
import torchaudio
from scipy.signal import get_window

from sot.cqt import CQT


class _BaseTransform(nn.Module):
    """Base class for spectral transforms."""

    def __init__(self, bin_position_scaling="normalized", device="cpu"):
        super().__init__()
        self.device = device
        self.bin_position_scaling = bin_position_scaling

    def get_bin_positions(self):
        """Returns the normalized positions of the frequency bins."""
        raise NotImplementedError

    def forward(self, x):
        """Takes a waveform and returns its spectral representation."""
        raise NotImplementedError


class STFT(_BaseTransform):
    """STFT transform module."""

    def __init__(
        self,
        fft_size=1024,
        hop_length=256,
        win_length=None,
        sr=None,
        window="flattop",
        bin_position_scaling="normalized",
        device="cpu",
    ):
        super().__init__(bin_position_scaling, device)
        self.fft_size = fft_size
        self.hop_size = hop_length
        self.sr = sr
        self.win_length = win_length if win_length else fft_size
        self.window = torch.tensor(
            get_window(window, self.win_length, fftbins=True),
            dtype=torch.float32,
        ).to(device)

    def get_bin_positions(self):
        """Returns normalized frequency bin positions for STFT."""
        if "normalized" in self.bin_position_scaling:
            return torch.linspace(0, 1, self.fft_size // 2 + 1)
        elif self.bin_position_scaling == "absolute":
            return torch.fft.rfftfreq(self.fft_size, d=1 / self.sr)

    def forward(self, x):
        """Computes the Short-Time Fourier Transform."""
        x = x.to(self.device)
        spec = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        return torch.abs(spec)


class MelSpectrogram(_BaseTransform):
    """Mel Spectrogram transform module."""

    def __init__(
        self,
        sr=22050,
        fft_size=1024,
        n_mels=128,
        hop_length=256,
        bin_position_scaling="normalized",
        device="cpu",
    ):
        super().__init__(bin_position_scaling, device)
        self.n_mels = n_mels
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=fft_size,
            n_mels=n_mels,
            hop_length=hop_length,
        ).to(self.device)

    def get_bin_positions(self):
        """Returns normalized frequency bin positions for Mel Spectrogram."""
        if self.bin_position_scaling == "normalized":
            return torch.linspace(0, 1, self.n_mels)
        else:
            raise NotImplementedError

    def forward(self, x):
        """Computes the Mel Spectrogram."""
        x = x.to(self.device)
        return self.transform(x)


class VQT(_BaseTransform):
    """CQT transform module."""

    def __init__(
        self,
        sr=22050,
        hop_length=512,
        fmin=32.7,
        fmax=None,
        n_bins=84,
        bins_per_octave=36,
        gamma=0.0,
        window="hann",
        device="cpu",
        bin_position_scaling="normalized",
    ):
        super().__init__(bin_position_scaling, device)

        self.transform = CQT(
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            n_bins=n_bins,
            gamma=gamma,
            window=window,
            bins_per_octave=bins_per_octave,
        ).to(self.device)
        self.n_bins = self.transform.n_bins

    def get_bin_positions(self):
        """Returns normalized frequency bin positions for CQT."""

        if self.bin_position_scaling == "normalized":
            freqs = torch.tensor(self.transform.frequencies, dtype=torch.float32)
            min_freq = freqs.min()
            max_freq = freqs.max()
            return (freqs - min_freq) / (max_freq - min_freq)
        elif self.bin_position_scaling == "absolute":
            return torch.tensor(self.transform.frequencies, dtype=torch.float32)
        elif self.bin_position_scaling == "normalized_linear":
            return torch.linspace(0, 1, self.n_bins, dtype=torch.float32)

    def forward(self, x):
        """Computes the Constant-Q Transform."""
        x = x.to(self.device).float()
        return torch.abs(self.transform(x))
