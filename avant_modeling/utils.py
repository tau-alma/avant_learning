import numpy as np
import scipy
import torch
import random
import matplotlib.pyplot as plt
import avant_modeling.config as config
from scipy.signal import savgol_filter
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


def plot_sample_data(data, sample_indices, out_file):
    # Plot some properties of the training data:
    fig, ax = plt.subplots(4, 2, figsize=(1000 / 100, 1000 / 100))
    fig.tight_layout()
    for i in range(4):
        for j in range(2):
            if i == 0:
                if j == 0:
                    ax[i, j].set_title("steer")
                    ax[i, j].hist(data[sample_indices, 7], 250)
                if j == 1:
                    ax[i, j].set_title("gas")
                    ax[i, j].hist(data[sample_indices, 8], 250)
            if i == 1:
                if j == 0:
                    ax[i, j].set_title("v_f")
                    ax[i, j].hist(data[sample_indices, 6], 250)
                if j == 1:
                    ax[i, j].set_title("omega")
                    ax[i, j].hist(data[sample_indices, 4], 250)
            if i == 2:
                if j == 0:
                    ax[i, j].set_title("beta")
                    ax[i, j].hist(data[sample_indices, 3], 250)
                if j == 1:
                    ax[i, j].set_title("dot_beta")
                    ax[i, j].hist(data[sample_indices, 5], 250)
            if i == 3:
                if j == 0:
                    ax[i, j].set_title("dot_dot_beta")
                    ax[i, j].hist(data[sample_indices, 9], 250)
                if j == 1:
                    ax[i, j].set_title("a_f")
                    ax[i, j].hist(data[sample_indices, 10], 250)
    plt.savefig(out_file)
    plt.close()


def plot_error_density(errors, name, out_file):
    num_errors = len(errors)
    df_errors = pd.DataFrame({
        'Error Type': [name] * num_errors + [name] * num_errors,
        'Dynamics model': ['Nominal'] * num_errors + ['Nominal + GP'] * num_errors,
        'Error': np.concatenate((
            errors[:, 0], errors[:, 1]
        ))
    })

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.set(font_scale=3.5)

    if name in ["omega_f", "dot_beta"]:
        xlabel = f"|{name}| error ($deg/s$)"
    elif name == "v_f":
        xlabel = f"|{name}| error ($m/s$)"
    else:
        xlabel = f"|{name}| error ($deg/s²$)"

    sns.kdeplot(data=df_errors, x='Error', hue='Dynamics model', ax=ax, fill=True)
    ax.set_title(' ', fontsize=40)
    ax.set_ylabel("Density", fontsize=40)
    ax.set_xlabel(xlabel, fontsize=40)
    ax.set_xlim(left=0, right=max(
            errors[:, 0].mean() + 2* errors[:, 0].std(),
            errors[:, 1].mean() + 2* errors[:, 1].std(),
        )
    )
    ax.tick_params(axis='both', which='major', labelsize=40)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Adjust layout
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def filter_data_batch(batch, data_idx_deriv_delay, output_dir=None):
    tmp_original = None
    tmp_filtered = None
    dt_ns = (batch[1:, 0] - batch[:-1, 0])
    window_size = int(1e9*config.savgol_p / np.mean(dt_ns))

    timestamps = batch[config.low_pass_window_size // 4: -config.low_pass_window_size // 4][:-1, 0].copy()

    for idx, deriv, delay, name in data_idx_deriv_delay:
        if delay == 0:
            selected = batch[:, idx].copy()
        if delay > 0:
            selected = batch[delay:, idx].copy()
        if delay < 0:
            selected = batch[:delay, idx].copy()

        if output_dir is not None:
            save_spectrum_plot(selected, 1e9 / np.mean(dt_ns), output_dir + "/data_spectrum_%s.png" % name)

        if name != "theta":
            selected = zero_phase_filtering(selected, 2, 1e9 / np.mean(dt_ns), config.low_pass_window_size, 3)
        else:
            selected = selected[config.low_pass_window_size // 4: -config.low_pass_window_size // 4]

        if deriv:
            savgol = 1e9*savgol_filter(
                selected, window_length=window_size, polyorder=config.savgol_d_k, deriv=1
            )[:-1] / dt_ns.mean()
            # Use finite difference derivative as reference:
            selected = 1e9 * (selected[1:] - selected[:-1]) / dt_ns.mean()
            selected = np.r_[selected, 0]
        else:
            if name != "theta":
                savgol = savgol_filter(
                    selected, window_length=window_size, polyorder=config.savgol_k, deriv=0
                )[:-1]
            else:
                savgol = selected[:-1]

        if delay > 0 or delay < 0:
            selected = np.r_[selected, np.zeros(np.abs(delay))]
            savgol = np.r_[savgol, np.zeros(np.abs(delay))]

        if tmp_original is None:
            tmp_original = selected[:-1]
            tmp_filtered = savgol
        else:
            tmp_original = np.c_[tmp_original, selected[:-1]]
            tmp_filtered = np.c_[tmp_filtered, savgol]

    return timestamps, tmp_original, tmp_filtered


def save_spectrum_plot(data, fs, path):
    """
    Save a frequency spectrum plot of the provided data.

    Parameters:
    - data: The input data array.
    - fs: The sampling frequency of the data.
    - path: Path (including filename) where the plot should be saved.
    """

    # Compute the FFT of the data
    spectrum = np.fft.fft(data)

    # Convert magnitude to dB scale and normalize
    magnitude_dB = 20 * np.log10(np.abs(spectrum / len(data)))

    # Frequency axis (upto Nyquist frequency)
    freqs = np.linspace(0, fs / 2, len(data) // 2)

    # Plot the magnitude in dB of the FFT (only the positive frequencies)
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, magnitude_dB[:len(data) // 2])
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)

    # Save the plot to the specified path
    plt.savefig(path)
    plt.close()


def zero_phase_filtering(data, cutoff_freq, fs, window_size=1000, order=4):
    """
    Zero-phase Butterworth filtering using a sliding window approach.

    Parameters:
    - data: The input data array.
    - cutoff_freq: The cutoff frequency for the Butterworth filter.
    - fs: The sampling frequency of the data.
    - window_size: Size of the sliding window.
    - order: The order of the Butterworth filter.

    Returns:
    - filtered_data: The filtered data.
    """

    # Create Butterworth filter coefficients
    b, a = scipy.signal.butter(order, cutoff_freq / (0.5 * fs), btype='low')

    # Number of valid points from each window
    valid_points = window_size // 2

    # Pre-allocate memory for filtered data
    filtered_data = np.zeros(len(data) - window_size + valid_points)

    for i in range(0, len(data) - window_size + 1):
        # Extract the current window from data
        window = data[i:i + window_size]

        # Forward filtering
        forward_filtered = scipy.signal.lfilter(b, a, window)

        # Backward filtering
        backward_filtered = scipy.signal.lfilter(b, a, forward_filtered[::-1])[::-1]

        # Extract valid points and store in filtered data
        filtered_data[i:i + valid_points] = backward_filtered[window_size // 4:window_size // 4 + valid_points]

    return filtered_data


# We can't just fit the GP to the full data since we will run out of GPU memory, so we want to select maximally distant samples from the data instead:
def farthest_point_sampling(points, num_samples, print_interval=5000):
    points = torch.from_numpy(points).cuda()
    num_points = points.shape[0]
    # Start with a random point
    first_index = torch.randint(num_points, (1,)).item()
    sampled_indices = [first_index]
    # Initialize the distance list with distances to the first point
    min_distances = torch.linalg.vector_norm(points - points[first_index], dim=1)
    for i in range(1, num_samples):
        if i % print_interval == 0:
            print(f"Progress: {i}/{num_samples} samples selected.")
        # Get the farthest point based on current min_distances
        farthest_point_idx = torch.argmax(min_distances).item()
        sampled_indices.append(farthest_point_idx)
        # Update the minimum distances with respect to the new farthest point
        distances_to_new_point = torch.linalg.vector_norm(points - points[farthest_point_idx], dim=1)
        min_distances = torch.min(min_distances, distances_to_new_point)
    return sampled_indices