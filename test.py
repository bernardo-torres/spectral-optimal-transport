import matplotlib.pyplot as plt
import numpy as np
import torch

from sot.features import STFT, VQT, MelSpectrogram
from sot.losses import MultiResolutionSOTLoss, Wasserstein1DLoss, wasserstein_1d

SAMPLE_RATE = 22050
DURATION = 1.0
FIXED_FREQ = 4000.0
SWEEP_FREQS = np.linspace(60, 8000, 200)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

t = torch.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), device=DEVICE)
fixed_sine = torch.sin(2 * torch.pi * FIXED_FREQ * t).unsqueeze(0)
fixed_sine_2 = torch.sin(2 * torch.pi * FIXED_FREQ * 1.2 * t).unsqueeze(0)

# Let's create a random signal and modulate it in time (shift in time) to test time-axis losses
np.random.seed(0)
# random signal is a sum of many harmonics
t2 = torch.linspace(
    0, 3 * DURATION, int(SAMPLE_RATE * 3 * DURATION), device=DEVICE
)  # Fix: 3x duration
random_signal = (
    sum(0.1 * torch.sin(2 * torch.pi * f * t2) for f in np.random.uniform(100, 8000, size=50))
    .unsqueeze(0)
    .to(DEVICE)
)

base_length = int(SAMPLE_RATE * DURATION * 0.6)  # 60% of full duration
t_base = torch.linspace(0, DURATION * 0.6, base_length, device=DEVICE)
base_random_signal = sum(
    0.1 * torch.sin(2 * torch.pi * f * t_base) for f in np.random.uniform(100, 8000, size=50)
)

# Create zero-padded version (full length with signal in the center)
full_length = int(SAMPLE_RATE * DURATION)
pad_size = (full_length - base_length) // 2
random_signal = torch.zeros(full_length, device=DEVICE)
random_signal[pad_size : pad_size + base_length] = base_random_signal
random_signal = random_signal.unsqueeze(0)  # Add batch dimension

# Create time-shifted versions by progressively shifting the padded signal
shift_amount = int(0.05 * SAMPLE_RATE)  # shift by 0.05 seconds
random_signals_shifted = []

for i in range(-5, 5):  # -5 to 4 shifts (10 total)
    shift = shift_amount * i

    # Roll the entire signal (including zero-padded regions)
    shifted_signal = torch.roll(random_signal, shifts=shift, dims=-1)
    random_signals_shifted.append(shifted_signal)


def plot_transforms(fixed_sig, sweep_sig, sr):
    """Computes and plots STFT, Mel, and CQT for two signals."""
    print("\n--- Plotting Spectral Transforms ---")
    fig, axs = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    fig.suptitle("Spectral Transforms Comparison", fontsize=16)

    transforms = {
        "STFT": STFT(device=DEVICE, sr=sr),
        "Mel Spectrogram": MelSpectrogram(sr=sr, device=DEVICE),
        "CQT": VQT(sr=sr, fmin=60, fmax=sr // 2, device=DEVICE),
        "VQT": VQT(sr=sr, fmin=60, fmax=sr // 2, device=DEVICE, gamma=7),
    }

    for i, (name, T) in enumerate(transforms.items()):
        fixed_spec = T(fixed_sig).squeeze().cpu().numpy()
        sweep_spec = T(sweep_sig).squeeze().cpu().numpy()

        for j, (spec, title) in enumerate(
            [
                (fixed_spec, f"Fixed Tone ({int(FIXED_FREQ/1000)}kHz)"),
                (sweep_spec, f"Sweep Tone ({int(SWEEP_FREQS[3]/1000)}kHz)"),
            ]
        ):
            ax = axs[i, j]
            im = ax.imshow(np.log1p(spec), aspect="auto", origin="lower", interpolation="none")
            ax.set_title(f"{name}: {title}")
            ax.set_xlabel("Time Frames")
            ax.set_ylabel("Frequency Bins")
            fig.colorbar(im, ax=ax)
    plt.show()


def run_loss_sweep(loss_modules, dim_desc):
    """Computes and plots Wasserstein loss across a frequency sweep."""
    print(f"\n--- Running Loss Sweep (Comparison along {dim_desc} axis) ---")
    losses = {name: [] for name in loss_modules.keys()}

    for freq in SWEEP_FREQS:
        sweep_sine = torch.sin(2 * torch.pi * freq * t).unsqueeze(0)
        for name, loss_fn in loss_modules.items():
            loss = loss_fn(fixed_sine, sweep_sine).item()
            losses[name].append(loss)

    plt.figure(figsize=(12, 7))
    for name, loss_values in losses.items():
        plt.plot(SWEEP_FREQS / 1000, loss_values, label=name)

    plt.axvline(
        x=FIXED_FREQ / 1000,
        color="r",
        linestyle="--",
        label=f"Fixed Freq ({int(FIXED_FREQ/1000)} kHz)",
    )
    plt.title(f"1D Wasserstein Distance vs. Sweep Frequency (Comparison: {dim_desc})")
    plt.xlabel("Sweep Signal Frequency (kHz)")
    plt.ylabel("Wasserstein Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_bin_scaling_effect():
    """Visualizes the effect of different bin_position_scaling settings for CQT."""
    print("\n--- Plotting Effect of Bin Position Scaling on CQT Loss ---")
    loss_modules = {
        "Normalized (Log Spacing)": Wasserstein1DLoss(
            transform="cqt",
            fmin=60,
            fmax=SAMPLE_RATE // 2,
            n_bins=120,
            device=DEVICE,
            sample_rate=SAMPLE_RATE,
            bin_position_scaling="normalized",
        ),
        "Normalized (Linear Spacing)": Wasserstein1DLoss(
            transform="cqt",
            fmin=60,
            fmax=SAMPLE_RATE // 2,
            n_bins=120,
            device=DEVICE,
            sample_rate=SAMPLE_RATE,
            bin_position_scaling="normalized_linear",
        ),
        # "Absolute (Hz)": Wasserstein1DLoss(
        #     transform="cqt",
        #     fmin=60,
        #     fmax=SAMPLE_RATE // 2,
        #     n_bins=120,
        #     device=DEVICE,
        #     sample_rate=SAMPLE_RATE,
        #     bin_position_scaling="absolute",
        # ),
    }
    run_loss_sweep(loss_modules, dim_desc="CQT Bin Position Scaling")


def plot_normalization_effect():
    """Visualizes the effect of `normalize` and `balanced` settings."""
    print("\n--- Plotting Effect of Normalization Schemes ---")
    loud_sine = fixed_sine * 2.0  # Double the amplitude
    quiet_sine = fixed_sine_2 * 0.5  # Half the amplitude

    loss_scenarios = {
        "Loud vs. Quiet (Balanced)": (
            loud_sine,
            quiet_sine,
            Wasserstein1DLoss(device=DEVICE, balanced=True, normalize=True),
        ),
        "Loud vs. Quiet (Unbalanced)": (
            loud_sine,
            quiet_sine,
            Wasserstein1DLoss(device=DEVICE, balanced=False, normalize=True),
        ),
        "Loud vs. Quiet (Unnormalized)": (
            loud_sine,
            quiet_sine,
            Wasserstein1DLoss(device=DEVICE, normalize=False),
        ),
        "Fixed vs. Quiet (Balanced)": (
            fixed_sine,
            quiet_sine,
            Wasserstein1DLoss(device=DEVICE, balanced=True, normalize=True),
        ),
    }

    results = {}
    for name, (sig1, sig2, loss_fn) in loss_scenarios.items():
        loss = loss_fn(sig1, sig2).item()
        results[name] = loss
        print(f"Loss for '{name}': {loss:.4f}")

    plt.figure(figsize=(10, 6))
    # bar plot with the loss values written on top of each bar
    plt.bar(results.keys(), results.values(), color="skyblue")
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom")

    plt.title("Effect of Normalization on Loss with Different Amplitudes")
    plt.ylabel("Wasserstein Loss")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.show()


def run_time_shift_comparison(loss_modules):
    """Compares time-shifted versions of the same signal using time-axis losses."""
    print("\n--- Running Time Shift Comparison ---")
    losses = {name: [] for name in loss_modules.keys()}

    # Use the original random signal as reference
    reference_signal = random_signal
    shift_labels = list(range(-5, 5))  # -5 to 4 shifts

    for i, shifted_signal in enumerate(random_signals_shifted):
        for name, loss_fn in loss_modules.items():
            loss = loss_fn(reference_signal, shifted_signal).item()
            losses[name].append(loss)

    plt.figure(figsize=(12, 7))
    for name, loss_values in losses.items():
        plt.plot(shift_labels, loss_values, label=name, marker="o")

    plt.axvline(x=0, color="r", linestyle="--", label="No Shift (Reference)")
    plt.title("1D Wasserstein Distance vs. Time Shift")
    plt.xlabel("Time Shift (× 0.05 seconds)")
    plt.ylabel("Wasserstein Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_quantile_functions():
    """Visualizes the quantile functions (inverse CDFs) of two spectra."""
    print("\n--- Plotting Quantile Functions ---")
    # Create two different signals
    low_freq_sine = torch.sin(2 * torch.pi * 1000 * t).unsqueeze(0)
    high_freq_sine = torch.sin(2 * torch.pi * 6000 * t).unsqueeze(0)

    loss_fn = Wasserstein1DLoss(return_quantiles=True, device=DEVICE, square_magnitude=True)

    # Get the quantile data from the loss function
    u_quantiles, v_quantiles, qs, _, _ = loss_fn(low_freq_sine, high_freq_sine)

    # Reshape for plotting (taking the first frame)
    qs = qs[0, 0].cpu().numpy()
    u_quantiles = u_quantiles[0, 0].cpu().numpy()
    v_quantiles = v_quantiles[0, 0].cpu().numpy()

    plt.figure(figsize=(12, 7))
    plt.plot(qs, u_quantiles, label="Low Freq Signal Quantiles (Inverse CDF)", marker=".")
    plt.plot(qs, v_quantiles, label="High Freq Signal Quantiles (Inverse CDF)", marker=".")

    # Fill the area between the curves to represent the Wasserstein distance
    plt.fill_between(
        qs,
        u_quantiles,
        v_quantiles,
        color="lightgray",
        alpha=0.5,
        label="Wasserstein Distance Area",
    )

    plt.title("Visualization of Quantile Functions for Two Spectra")
    plt.xlabel("Quantile (Cumulative Probability)")
    plt.ylabel("Frequency Bin Position (Normalized)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    sweep_sine_sample = torch.sin(2 * torch.pi * SWEEP_FREQS[3] * t).unsqueeze(0)
    plot_transforms(fixed_sine, sweep_sine_sample, SAMPLE_RATE)

    loss_modules_freq = {
        "STFT (p=2, rooted)": Wasserstein1DLoss(
            transform="stft",
            dim=-1,
            device=DEVICE,
            sample_rate=SAMPLE_RATE,
            p=2,
            apply_root=True,
            square_magnitude=True,
        ),
        "STFT (p=2, unrooted)": Wasserstein1DLoss(
            transform="stft",
            dim=-1,
            device=DEVICE,
            sample_rate=SAMPLE_RATE,
            p=2,
            apply_root=False,
            square_magnitude=True,
        ),
        "Multi-Resolution STFT": MultiResolutionSOTLoss(
            transform="stft",
            dim=-1,
            device=DEVICE,
            sample_rate=SAMPLE_RATE,
            fft_sizes=[512, 1024, 2048],
            hop_lengths=[128, 256, 512],
            square_magnitude=True,
        ),
        "Mel ": Wasserstein1DLoss(
            transform="mel",
            dim=-1,
            device=DEVICE,
            sample_rate=SAMPLE_RATE,
            n_mels=128,
            square_magnitude=False,
        ),
        "CQT (Normalized Log)": Wasserstein1DLoss(
            transform="cqt",
            fmin=60,
            n_bins=120,
            dim=-1,
            device=DEVICE,
            sample_rate=SAMPLE_RATE,
            fmax=SAMPLE_RATE // 2,
        ),
        "VQT (Normalized Log)": Wasserstein1DLoss(
            transform="cqt",
            fmin=60,
            n_bins=120,
            dim=-1,
            gamma=7.0,
            device=DEVICE,
            sample_rate=SAMPLE_RATE,
            fmax=SAMPLE_RATE // 2,
        ),
    }
    run_loss_sweep(loss_modules_freq, dim_desc="Frequency")

    loss_modules_time = {
        "STFT Loss (Time Axis)": Wasserstein1DLoss(
            transform="stft", dim=-2, device=DEVICE, sample_rate=SAMPLE_RATE
        ),
        "CQT Loss (Time Axis)": Wasserstein1DLoss(
            transform="cqt",
            fmin=60,
            n_bins=120,
            dim=-2,
            device=DEVICE,
            sample_rate=SAMPLE_RATE,
            fmax=SAMPLE_RATE // 2,
        ),
    }
    run_time_shift_comparison(loss_modules_time)

    plot_bin_scaling_effect()

    plot_normalization_effect()

    plot_quantile_functions()
