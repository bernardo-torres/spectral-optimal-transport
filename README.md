# Spectral Optimal Transport Losses for PyTorch 


[![PyPI Version](https://img.shields.io/pypi/v/sot-loss)](https://pypi.org/project/sot-loss/) [![Paper (arXiv)](https://img.shields.io/badge/arXiv-2312.14507-b31b1b.svg)](https://arxiv.org/abs/2312.14507)

This repository contains an implementation of Spectral Optimal Transport (SOT) loss functions for PyTorch, with a pip-installable package `sot-loss`. SOT loss functions are differentiable spectral losses which compare the spectra of two audio signals using optimal transport principles. These loss functions can be used for training neural networks in audio processing tasks, particularly those involving DDSP. It can also be used more generally as a metric for audio signal comparison. 

<table>
  <tr>
    <td>
      <p align="center"><b>SOT does this  <br>                           </b></p>
      <img src="figures/poster_spectra_horizontal_transport_lines.png" width="380" />
    </td>
    <td>
      <p align="center"><b>Multi-Scale Spectral loss and others do this</b></p>
      <img src="figures/poster_spectra_vertical.png" width="380" />
    </td>
  </tr>
</table>

## Installation

You can install the `sot-loss` package using pip:

```bash
pip install sot-loss
```

## Usage

The primary components of this package are the `Wasserstein1DLoss` and `MultiResolutionSOTLoss` classes, which can be used as PyTorch loss functions. Here is a basic example of how to use the `Wasserstein1DLoss`:

```python
import torch
from sot import Wasserstein1DLoss

# Create some dummy audio signals
x = torch.randn(4, 16000)
y = torch.randn(4, 16000)

# Initialize the SOT loss
sot_loss = Wasserstein1DLoss(transform='stft', 
                             fft_size=2048,
                             hop_length=512, 
                             sample_rate=16000, 
                             window='flattop', 
                             square_magnitude=True)

# Compute the loss
loss = sot_loss(x, y)
print(loss)
```

Using your own mapping audio -> 2D representation:

```python
x_spec = custom_transform(x) # batch, channels, time
y_spec = custom_transform(y) 
x_positions = get_custom_positions(x_spec) # channels
y_positions = get_custom_positions(y_spec)

sot_loss = Wasserstein1DLoss(transform='identity',
                             # other non-transform parameters can go here
                             balanced=True,
                             normalize=True,
                             )
loss = sot_loss(x_spec, y_spec, x_positions=x_positions, y_positions=y_positions)
print(loss)
```


## Advanced Usage

The `Wasserstein1DLoss` and `MultiResolutionSOTLoss` classes offer a range of parameters to customize the spectral representation and the loss calculation.

### Spectral Transform Parameters

These parameters are available in both `Wasserstein1DLoss` and `MultiResolutionSOTLoss`.

Transform parameters (if using built-in transforms):
| Argument | Type | Default | Description |
|---|---|---|---|
| `transform` | str | `'stft'` | The spectral transform to use. One of `'stft'`, `'mel'`, `'cqt'`, or `'identity'`. |
| `fft_size`, `hop_length`, `win_length` | int | `1024`, `256`, `None`  | Your typical STFT parameters. |
| `window`| str | `'flattop'`| The window function to use for STFT and CQT. |
| `n_mels`| int | `128` | Number of Mel bins for the Mel spectrogram. |
| `n_bins`, `bins_per_octave`, `fmin`, `fmax`, `sample_rate` | int, int, float, float | `84`, `36`, `32.7`, `None`, `22050` | CQT parameters. |
| `gamma` | int | `0` | VQT parameter which reduces kernel lengths for low frequencies. `0` for traditional CQT (see [This paper](https://transactions.ismir.net/articles/10.5334/tismir.251)) . |
| `bin_position_scaling` | str | `'normalized'` | Defines how the ground distance for the Wasserstein calculation is measured. Affects how the bin positions for the transforms are calculated. One of `'normalized'`, `'normalized_linear'`, or `'absolute'`. |

Loss parameters (applies even if using custom transforms):

| Argument | Type | Default | Description |
|---|---|---|---|
| `square_magnitude` | bool | `False` | If `True`, computes the loss on the squared magnitude of the spectrum (power). |
| `dim` | int | `-1` | The dimension along which to compute the Wasserstein distance. `-1` for frequency, `-2` for time. |
| `normalize` | bool | `True` | If `True`, normalizes the spectral magnitudes to sum to 1, treating them as probability distributions. |
| `balanced` | bool | `True` | If `True` and `normalize` is `True`, both spectra are normalized to sum to 1 independently. If `False` and `normalize` is `True`, the second spectrum is scaled relative to the first. |
| `p` | int | `2` | The order of the Wasserstein distance. |
| `quantile_lowpass` | bool | `False` | If `True`, applies a frequency cutoff by zeroing out distances for quantiles above 1.0. This is useful when `balanced` is `False`. |



The `MultiResolutionSOTLoss` combines multiple `Wasserstein1DLoss` instances, each with a different set of STFT parameters.

| Argument | Type | Default | Description |
|---|---|---|---|
| `fft_sizes` | list | `[1024, 2048, 512]` | A list of FFT sizes to use for each resolution. |
| `hop_lengths` | list | `[256, 512, 128]` | A list of hop lengths to use for each resolution. |
| `win_lengths` | list | `None` | A list of window lengths to use for each resolution. If `None`, defaults to `fft_sizes`. |


## About the Paper

This is the also the official repository for the paper "[Unsupervised Harmonic Parameter Estimation Using Differentiable DSP and Spectral Optimal Transport.](https://arxiv.org/abs/2312.14507)", by *Bernardo Torres, Geoffroy Peeters, and Gaël Richard*. Check out the [poster here](https://bernardo-torres.github.io/documents/Torres_ICASSP_2024_poster.pdf).


For repoducing the results from the paper, please check out the [paper branch](https://github.com/bernardo-torres/spectral-optimal-transport/tree/paper).



## Citation


If you find our work useful or use it in your research, you can cite it using:

```bibtex
@inproceedings{torres2024unsupervised,
  title={Unsupervised harmonic parameter estimation using differentiable DSP and spectral optimal transport},
  author={Torres, Bernardo and Peeters, Geoffroy and Richard, Ga{\"e}l},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1176--1180},
  year={2024},
  organization={IEEE}
}

```
