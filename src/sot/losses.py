"""Implements spectral optimal transport losses for comparing audio signals.

This module provides PyTorch loss functions based on the 1D Wasserstein distance
(Optimal Transport) calculated on various spectral representations of audio.
The primary loss, `Wasserstein1DLoss`, can operate on STFT, Mel Spectrogram,
or CQT representations. A multi-resolution version is also available to
capture spectral differences at different time-frequency trade-offs.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import get_window

from sot.features import STFT, VQT, MelSpectrogram


def quantile_function(qs, cws, xs):
    """Computes the quantile function (inverse CDF) for a discrete distribution.

    This is a helper function used by `wasserstein_1d` to find the value of a
    distribution at specific quantile levels.

    Args:
        qs (torch.Tensor): The quantile levels to evaluate, shape (..., k).
        cws (torch.Tensor): The cumulative weights of the distribution, shape (..., n).
        xs (torch.Tensor): The values (positions) of the distribution, shape (..., n).

    Returns:
        torch.Tensor: The values of the distribution corresponding to the given
            quantile levels, shape (..., k).
    """
    n = xs.shape[-1]
    idx = torch.searchsorted(cws, qs)
    return torch.take_along_dim(xs, torch.clamp(idx, 0, n - 1), dim=-1)


def wasserstein_1d(
    u_values,
    v_values,
    u_weights=None,
    v_weights=None,
    p=1,
    require_sort=True,
    return_quantiles=False,
    limit_quantile_range=False,
):
    """Approximates the 1D Wasserstein distance between two distributions by a sum of distances between quantiles.
    We assume  (u_weights, v_weights)  belong to the space of probability vectors, $i.e.$ $u_weights \in \Sigma_n$ and
    $v_weights \in \Sigma_m$, for $\Sigma_n = \left\{\mathbf{a} \in \mathbb{R}^n_+ ; \sum_{i=1}^n \mathbf{a}_i = 1 \right\}$.
    That means the weights are normalized to sum to 1 and are non-negative.

    The Wasserstein distance between two one dimensional distributions can be expressed in closed form as [1, prop. 2.17, 2, Remark 2.30]:

     \mathcal{W}_p(\alpha, \beta)^{p} =  \int_0^1 \left| F^{-1}_{\alpha}(r) - F^{-1}_{\beta}(r) \right|^p dr

    where F^{-1}_{\alpha} is the quantile function, or inverse CDF of \alpha.

    We approximate this integral by a sum of distances between quantiles as it's done in POT [3]:

    \mathcal{W}_p(\alpha, \beta)^{p} =  \sum_{i=1}^n \left| F^{-1}_{\alpha}(r_i) - F^{-1}_{\beta}(r_i) \right|^p (r_i - r_{i-1}),

    where r_i is the ith quantile of the ordered set of quantiles of \alpha and \beta. We use the step function to compute and inverse the
        CDF by "holding" the value of the quantile constant between quantiles.

    [1] F. Santambrogio, “Optimal transport for applied mathematicians,” Birkäuser, NY, vol. 55, no. 58–63, p. 94, 2015.
    [2] G. Peyré and M. Cuturi, “Computational optimal transport,” Foundations and Trends in Machine Learning, vol. 11, no. 5–6, pp. 355–607, 2019.
    [3] R. Flamary et al., “POT: Python optimal transport,” Journal of Machine Learning Research, vol. 22, no. 78, pp. 1–8, 2021.

    Code inspired by POT's implementation: https://pythonot.github.io/_modules/ot/lp/solver_1d.html#wasserstein_1d

    Args:
        u_values: Tensor of shape (batch, n) containing the locations of weights values of the first distribution.
        u_weights: Tensor of shape (batch, n) containing the weights of the first distribution.
        v_values: Tensor of shape (batch, m) containing the locations of weights values of the second distribution.
        v_weights: Tensor of shape (batch, m) containing the weights of the second distribution.
        p: Order of the Wasserstein distance.
        require_sort: If True, sort u_values and v_values before computing the loss.
        return_quantiles: If True, return the quantiles of u and v.
        limit_quantile_range: If True, set the distance to 0 if the quantile is greater than 1 (which can happen if non-normalized weights are used as input).
    Returns:
        The 1D Wasserstein distance between the two distributions.

    """

    assert p >= 1, f"The OT loss is only valid for p>=1, {p} was given"

    n = u_values.shape[1]
    m = v_values.shape[1]

    if u_weights is None:
        u_weights = torch.full(
            u_values.shape, 1.0 / n, device=u_values.device, dtype=u_values.dtype
        )

    if v_weights is None:
        v_weights = torch.full(
            v_values.shape, 1.0 / m, device=v_values.device, dtype=v_values.dtype
        )

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, 1)
        v_values, v_sorter = torch.sort(v_values, 1)
        u_weights = torch.gather(u_weights, 1, u_sorter)
        v_weights = torch.gather(v_weights, 1, v_sorter)

    u_cumweights = torch.cumsum(u_weights, 1)
    v_cumweights = torch.cumsum(v_weights, 1)

    qs = torch.sort(torch.cat((u_cumweights, v_cumweights), 1), 1)[0]
    # qs = torch.sort(torch.concatenate((u_cumweights, v_cumweights), 1), 1)
    u_quantiles = quantile_function(qs, u_cumweights, u_values)
    v_quantiles = quantile_function(qs, v_cumweights, v_values)
    if return_quantiles:
        return u_quantiles, v_quantiles, qs, u_cumweights, v_cumweights
    qs = torch.nn.functional.pad(qs, pad=(1, 0))

    # qs = torch.nn.functional.pad(qs, (1, 0), mode='constant', value=0)
    delta = qs[..., 1:] - qs[..., :-1]
    # Set to 0 if qs > 1
    if limit_quantile_range:
        delta = torch.where(qs[..., 1:] > 1, torch.zeros_like(delta), delta)

    diff_quantiles = torch.abs(u_quantiles - v_quantiles)

    if p == 1:
        return torch.sum(delta * diff_quantiles, 1)
    return torch.sum(delta * diff_quantiles.pow(p), 1)


class _BaseSOTLoss(nn.Module):
    """Base class for Spectral Optimal Transport (SOT) losses."""

    def __init__(
        self,
        transform: str = "stft",
        fft_size=1024,
        hop_length=256,
        win_length=None,
        window="flattop",
        n_mels=128,
        n_bins=250,
        bins_per_semitone=3,
        gamma=0,
        fmin=32.7,
        fmax=None,
        sample_rate=22050,
        bin_position_scaling="normalized",  # 'normalized', 'absolute', 'normalized_linear'
        square_magnitude=False,
        dim=-1,
        eps=1e-8,
        device="cpu",
        return_quantiles=False,
        reduce=True,
        **kwargs,
    ):
        """Initializes the base loss module and its associated spectral transform.

        Args:
            transform (str, optional): The spectral transform to use. One of
                {'stft', 'mel', 'cqt', 'identity'}. Defaults to 'stft'.
            fft_size (int, optional): FFT size for STFT and Mel transforms.
                Defaults to 1024.
            hop_length (int, optional): Hop length for the transform. Defaults to 256.
            win_length (int, optional): Window length for the transform. If None,
                defaults to `fft_size`. Defaults to None.
            window (str, optional): The window function to use for STFT and CQT.
                Defaults to "flattop".
            n_mels (int, optional): Number of Mel bins for the Mel spectrogram.
                Defaults to 128.
            n_bins (int, optional): Number of frequency bins for the CQT.
                Defaults to 250.
            bins_per_semitone (int, optional): Number of CQT bins per semitone.
                Used to calculate `bins_per_octave`. Defaults to 3.
            gamma (int, optional): VQT parameter which reduces kernel lengths for
                low frequencies. Defaults to 0 (traditional CQT).
            fmin (float, optional): Minimum frequency for CQT. Defaults to 32.7.
            fmax (float, optional): Maximum frequency for CQT. If None, it is
                determined by `n_bins`. Defaults to None.
            sample_rate (int, optional): The sample rate of the input audio.
                Defaults to 22050.
            bin_position_scaling (str, optional): Defines how the ground distance
                for the Wasserstein calculation is measured.
                - 'normalized': Preserves the true spacing
                  of bins, scaled to the [0, 1] range.
                - 'normalized_linear': Assumes linear spacing of bins, scaled to
                  the [0, 1] range. This will not reflect the actual frequency spacing
                    of log-frequency transforms like CQT or Mel and will give higher weight
                    to lower frequencies.
                - 'absolute': Uses the raw frequency values in Hz. This will give higher
                 loss values than normalized versions.
                Defaults to 'normalized'.
                        square_magnitude (bool, optional): If True, computes the loss on the
                squared magnitude of the spectrum (power). Defaults to False.
            dim (int, optional): The dimension along which to compute the
                Wasserstein distance. -1 for frequency, -2 for time.
                Defaults to -1.
            eps (float, optional): A small epsilon value for numerical stability.
                Defaults to 1e-8.
            device (str, optional): The compute device ('cpu' or 'cuda').
                Defaults to "cpu".
            reduce (bool, optional): If True, returns the mean of the loss over
                the batch. Otherwise, returns the loss for each item.
                Defaults to True.

        Raises:
            ValueError: If an unknown `transform` type is provided.
        """
        super().__init__()

        if transform == "stft":
            self.transform_module = STFT(
                fft_size=fft_size,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                sr=sample_rate,
                bin_position_scaling=bin_position_scaling,
                device=device,
            )
        elif transform == "mel":
            self.transform_module = MelSpectrogram(
                sr=sample_rate,
                n_mels=n_mels,
                fft_size=fft_size,
                hop_length=hop_length,
                bin_position_scaling=bin_position_scaling,
                device=device,
            )
        elif transform == "cqt" or transform == "vqt":
            self.transform_module = VQT(
                sr=sample_rate,
                hop_length=hop_length,
                fmin=fmin,
                fmax=fmax,
                n_bins=n_bins,
                bins_per_octave=bins_per_semitone * 12,
                gamma=gamma,
                window=window,
                bin_position_scaling=bin_position_scaling,
                device=device,
            )
        elif transform is None or transform == "identity":
            print(f"Unknown transform {transform}, defaulting to identity")
            self.transform_module = nn.Identity()
        else:
            raise ValueError(f"Unknown transform {transform}")

        self.device = device
        self.dim = dim
        if not isinstance(self.transform_module, nn.Identity) and dim == -1:
            # Register buffer ensures the tensor is moved to the correct device with the model
            self.register_buffer("bin_positions", self.transform_module.get_bin_positions())
        else:
            self.bin_positions = None
        self.eps = eps
        self.square_magnitude = square_magnitude
        self.reduce = reduce
        self.return_quantiles = return_quantiles
        self.to(device)

    def forward(self, x, y, x_positions=None, y_positions=None):
        """Computes the loss between two audio signals.

        First, the spectral representation of each signal is computed. Then, the
        loss is calculated on these spectra. Handles mono and stereo inputs.

        Args:
            x (torch.Tensor): The first input audio signal, shape (batch, samples)
                or (batch, channels, samples).
            y (torch.Tensor): The second input audio signal, with the same shape as x.
            x_positions (torch.Tensor, optional): Custom positions for the bins of x.
                If None, uses the default positions from the transform.
                Defaults to None.
            y_positions (torch.Tensor, optional): Custom positions for the bins of y.
                If None, uses the default positions from the transform.
                Defaults to None.

        Returns:
            torch.Tensor: The computed loss value.
        """
        x = x.to(self.device)
        y = y.to(self.device)

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)

        self.was_stereo = False
        if x.ndim == 3:
            # Stereo, we move to batch dimension
            x = x.view(-1, x.shape[-1])
            self.was_stereo = True
            y = y.view(-1, y.shape[-1])

        x_spec = self.transform_module(x)
        y_spec = self.transform_module(y)

        # We put channel in the last dimension always
        x_spec = x_spec.permute(0, 2, 1)
        y_spec = y_spec.permute(0, 2, 1)

        if x_spec.ndim == 2:
            x_spec = x_spec.unsqueeze(1)
        if y_spec.ndim == 2:
            y_spec = y_spec.unsqueeze(1)

        if self.square_magnitude:
            x_spec = x_spec**2
            y_spec = y_spec**2

        x_pos = x_positions if x_positions is not None else self.bin_positions
        y_pos = y_positions if y_positions is not None else self.bin_positions

        if x_pos is None and self.dim == -2 or self.dim == 1:
            # We consider the time dimension, we create a dummy position vector
            x_pos = torch.arange(x_spec.shape[1], device=self.device).float()
            y_pos = torch.arange(y_spec.shape[1], device=self.device).float()

        loss = self.loss(x_spec, y_spec, x_positions=x_pos, y_positions=y_pos)
        if self.return_quantiles:
            return loss
        if loss.shape[0] == x.shape[0] and self.was_stereo:
            # We had stereo input and , recover the original shape
            loss = loss.view(-1, 2)
        return self._reduce_loss(loss)

    def _reduce_loss(self, loss):
        return torch.mean(loss) if self.reduce else loss

    def loss(self, x_values, y_values, x_positions=None, y_positions=None, p=1):
        """Abstract method for computing the loss on spectral representations.

        This method must be implemented by all subclasses.
        """
        raise NotImplementedError


class Wasserstein1DLoss(_BaseSOTLoss):
    """Computes the 1D Wasserstein distance between the spectra of two audio signals along a given dimension."""

    def __init__(
        self,
        transform: str = "stft",
        normalize=True,
        balanced=True,
        p=2,
        return_quantiles=False,
        quantile_lowpass=False,
        eps=1e-8,
        dim=-1,
        apply_root=False,  # Wether to return Wp_p or Wp. Wp^p might be more stable for optimization
        device="cpu",
        **kwargs,
    ):
        """
        Args:
            transform (str, optional): The spectral transform to use.
                Defaults to 'stft'.
            normalize (bool, optional): If True, normalizes the spectral magnitudes
                to sum to 1, treating them as probability distributions.
                Defaults to True.
            balanced (bool, optional): If True and `normalize` is True, both spectra
                are normalized to sum to 1 independently. If False and `normalize` is True,
                the second spectrum is scaled relative to the first. Defaults to True.
            p (int, optional): The order of the Wasserstein distance (e.g., p=1 for
                Earth Mover's Distance, p=2 for a quadratic cost). Defaults to 2.
            return_quantiles (bool, optional): If True, the forward pass returns
                intermediate quantile information instead of the loss.
                Defaults to False.
            quantile_lowpass (bool, optional): If True, applies a frequency cutoff
                by zeroing out distances for quantiles above 1.0. This is useful
                when `balanced` is False. Defaults to False.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-8.
            dim (int, optional): The dimension along which to compute the distance.
                -1 for frequency, -2 for time. Defaults to -1.
            apply_root (bool, optional): If True, applies the p-th root to the result
                to get the true W_p distance. If False, returns W_p^p, which can be
                more stable for optimization. Defaults to False.
            device (str, optional): The compute device. Defaults to "cpu".
        """
        super().__init__(
            transform, eps=eps, return_quantiles=return_quantiles, device=device, dim=dim, **kwargs
        )
        self.p = p
        self.normalize = normalize
        self.balanced = balanced
        self.return_quantiles = return_quantiles
        self.quantile_lowpass = quantile_lowpass
        self.apply_root = apply_root
        self.dim = dim

    def loss(self, x_spec, y_spec, x_positions=None, y_positions=None):
        """Calculates the 1D Wasserstein distance on the input spectra.

        This method handles normalization and shaping of the spectral tensors before
        passing them to the core `wasserstein_1d` function.

        Args:
            x_spec (torch.Tensor): The spectral representation of the first signal.
            y_spec (torch.Tensor): The spectral representation of the second signal.
            x_positions (torch.Tensor): The positions of the bins for x_spec.
            y_positions (torch.Tensor): The positions of the bins for y_spec.

        Returns:
            torch.Tensor or list: The Wasserstein distance. If `return_quantiles`
                is True, returns a list of intermediate quantile tensors.
        """
        original_shape = x_spec.shape[:-1]
        # Let's put the dimension of interest at the end
        if self.dim != -1 and self.dim != x_spec.ndim - 1:
            x_spec = x_spec.transpose(self.dim, -1)
            y_spec = y_spec.transpose(self.dim, -1)
        # Normalize magnitudes to be distributions
        total_mass_x = torch.sum(x_spec, dim=-1, keepdim=True) + self.eps
        total_mass_y = torch.sum(y_spec, dim=-1, keepdim=True) + self.eps
        if self.normalize:
            x_spec = x_spec / total_mass_x
            if self.balanced:
                # Both masses sum to 1
                y_spec = y_spec / total_mass_y
            else:
                # X sums to 1, y is normalized to have a proportional mass
                y_spec = y_spec / total_mass_x
        elif self.balanced:
            # masses do not sum to 1, but we still want them to have the same mass
            y_spec = y_spec / total_mass_y * total_mass_x

        x_spec = x_spec.reshape(-1, x_spec.shape[-1])  # (batch, n_bins)
        y_spec = y_spec.reshape(-1, y_spec.shape[-1])  # (

        if x_positions.ndim == 1:
            x_positions = x_positions.expand_as(x_spec)
        if y_positions.ndim == 1:
            y_positions = y_positions.expand_as(y_spec)

        loss = wasserstein_1d(
            x_positions,
            y_positions,
            x_spec,
            y_spec,
            p=self.p,
            require_sort=True,
            return_quantiles=self.return_quantiles,
            limit_quantile_range=self.quantile_lowpass,
        )

        if self.apply_root and not self.return_quantiles:
            if self.p == 1:
                pass
            elif self.p == 2:
                loss = torch.sqrt(loss + self.eps)
            else:
                loss = loss.pow(1.0 / self.p)
        if self.return_quantiles:
            loss = [l.reshape(original_shape + (-1,)) for l in loss]
            return loss

        return loss


class MultiResolutionSOTLoss(nn.Module):
    """Computes a Spectral Optimal Transport (SOT) loss at multiple resolutions.

    This module combines multiple `Wasserstein1DLoss` instances, each with a
    different set of STFT parameters (e.g., FFT sizes), and sums their outputs.
    """

    def __init__(
        self,
        transform: str = "stft",
        fft_sizes=[1024, 2048, 512],
        hop_lengths=[256, 512, 128],
        win_lengths=None,
        **kwargs,
    ):
        super().__init__()
        SOTLoss = Wasserstein1DLoss  # For now only Wasserstein1DLoss is implemented

        self.loss_modules = nn.ModuleList()
        if win_lengths is None:
            win_lengths = fft_sizes
        for fft_size, hop, win in zip(fft_sizes, hop_lengths, win_lengths):
            # Pass all loss-specific kwargs to the Wasserstein1DLoss constructor
            self.loss_modules.append(
                SOTLoss(
                    transform=transform,
                    fft_size=fft_size,
                    hop_length=hop,
                    win_length=win,
                    **kwargs,
                )
            )

    def forward(self, x, y, **kwargs):
        loss = 0.0
        for f in self.loss_modules:
            loss += f(x, y, **kwargs)
        return loss
