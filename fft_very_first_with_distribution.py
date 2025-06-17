import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import os

# Define the output directory
OUTPUT_DIR = 'fft_and_polar_results_color'  # Updated output directory name


def ensure_output_dir():
    """Ensure the output directory exists"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}/")


def load_accelerometer_data(csv_file):
    """Load accelerometer data from CSV file and convert to standard units"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    required_cols = ['X', 'Y', 'Z', 'Encoder']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file must contain {', '.join(required_cols)} columns.")
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f"Missing columns: {', '.join(missing_cols)}")
        return pd.DataFrame()

    x_milli = df['X'].values
    y_milli = df['Y'].values
    z_milli = df['Z'].values - 1000
    conversion_factor = 1e-3
    df['X'] = x_milli * conversion_factor
    df['Y'] = y_milli * conversion_factor
    df['Z'] = z_milli * conversion_factor
    print(f"Converted X, Y, Z acceleration data to standard units (factor: {conversion_factor})")
    if 'Encoder' in df.columns:
        print(f"Encoder data loaded. Min: {df['Encoder'].min()}, Max: {df['Encoder'].max()}")
    print(f"Loaded {len(df)} samples")
    return df


def compute_fft(signal_data, sample_rate):
    if len(signal_data) == 0: return np.array([]), np.array([])
    windowed_signal = signal_data * signal.windows.hann(len(signal_data))
    fft_result = fft(windowed_signal)
    freqs = fftfreq(len(signal_data), 1.0 / sample_rate)
    magnitude = np.abs(fft_result) / len(signal_data)
    positive_indices = freqs >= 0
    freqs = freqs[positive_indices]
    magnitude = magnitude[positive_indices]
    if len(magnitude) > 1: magnitude[1:] = magnitude[1:] * 2
    return freqs, magnitude


def find_peak_frequencies(freqs, magnitude, num_peaks=5):
    if len(magnitude) <= 1: return np.array([]), np.array([])
    search_freqs, search_magnitude, offset = freqs, magnitude, 0
    if freqs[0] == 0 and len(freqs) > 1:
        search_freqs, search_magnitude, offset = freqs[1:], magnitude[1:], 1
    elif freqs[0] == 0 and len(freqs) == 1:
        return np.array([]), np.array([])
    if len(search_magnitude) == 0: return np.array([]), np.array([])
    actual_num_peaks = min(num_peaks, len(search_magnitude))
    if actual_num_peaks == 0: return np.array([]), np.array([])
    indices = np.argsort(search_magnitude)[-actual_num_peaks:] + offset
    sorted_indices_by_freq = sorted(indices, key=lambda i: freqs[i])
    return freqs[sorted_indices_by_freq], magnitude[sorted_indices_by_freq]


def plot_time_domain(df, output_dir, sample_rate_for_time_axis):
    output_file = os.path.join(output_dir, "time_domain_XYZ.png")
    plt.figure(figsize=(12, 9))
    timestamps = np.arange(len(df)) / sample_rate_for_time_axis  # Default
    time_source_info = f"index-based time axis with sample rate: {sample_rate_for_time_axis} Hz."
    if 'Timestamp' in df.columns:
        try:
            numeric_timestamps = pd.to_numeric(df['Timestamp'], errors='coerce')
            if not numeric_timestamps.isnull().all():
                timestamps = (numeric_timestamps - numeric_timestamps.iloc[0]) / 1000.0
                time_source_info = "'Timestamp' column (converted to seconds)."
        except Exception as e:
            print(f"Could not use 'Timestamp' for time axis ({e}), falling back.")
    elif 'Tick' in df.columns:
        try:
            timestamps = pd.to_numeric(df['Tick'], errors='coerce').values / sample_rate_for_time_axis
            time_source_info = f"'Tick' column with sample rate: {sample_rate_for_time_axis} Hz."
        except Exception as e:
            print(f"Could not use 'Tick' for time axis ({e}), falling back to index-based.")
    print(f"Using {time_source_info} for time domain plot.")

    for i, axis in enumerate(['X', 'Y', 'Z']):
        plt.subplot(3, 1, i + 1)
        plt.plot(timestamps, df[axis].fillna(0), label=f'{axis}-axis', color=['r', 'g', 'b'][i], alpha=0.7)
        plt.title(f'{axis}-axis Acceleration')
        plt.ylabel('Acceleration (units)')
        plt.grid(True)
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Time domain plot saved as {output_file}")
    plt.show()
    plt.close()


def plot_fft_results(axis_name, freqs, magnitude, peak_freqs, peak_mags, output_dir):
    output_file = os.path.join(output_dir, f"fft_results_{axis_name}.png")
    plt.figure(figsize=(12, 6))
    if len(freqs) == 0:
        print(f"Warning: No FFT data for {axis_name}-axis.")
    else:
        plt.plot(freqs, magnitude, 'b-', alpha=0.7)
    for f, m in zip(peak_freqs, peak_mags): plt.plot(f, m, 'ro'); plt.text(f, m * 1.1, f"{f:.2f} Hz", ha='center')
    plt.title(f'FFT Analysis of {axis_name}-axis Acceleration (Full Spectrum)')
    plt.xlabel('Frequency (Hz)');
    plt.ylabel('Magnitude');
    plt.grid(True)
    if len(freqs) > 0: plt.xlim(left=0, right=freqs[-1] if len(freqs) > 0 else None)
    if len(magnitude) > 0: plt.ylim(bottom=0)
    plt.tight_layout();
    plt.savefig(output_file, dpi=300);
    print(f"FFT plot for {axis_name}-axis saved as {output_file}");
    plt.show();
    plt.close()


def plot_vibration_polar(axis_name, df, output_dir):
    """Plots vibration for a given axis against encoder angle on a polar chart, colored by sample index."""
    if 'Encoder' not in df.columns:
        print(f"Error: 'Encoder' column not found. Cannot create polar plot for {axis_name}-axis.")
        return
    if df['Encoder'].isnull().all():
        print(f"Warning: 'Encoder' column all NaN for {axis_name}-axis. Skipping polar plot.")
        return

    output_file = os.path.join(output_dir, f"polar_vibration_color_{axis_name}.png")
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})

    theta = np.deg2rad(df['Encoder'].fillna(0).values)
    r = df[axis_name].fillna(0).values

    # Use a sequential array for color mapping (e.g., sample index)
    # This will show the progression of data over "time" (sample sequence)
    colors = df.index

    # Scatter plot with color based on sample index
    scatter = ax.scatter(theta, r, c=colors, cmap='viridis', alpha=0.6, s=10)  # s is marker size

    # Add a colorbar
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.1)
    cbar.set_label('Sample Index (Time Progression)')

    ax.set_title(f'{axis_name}-axis Vibration vs. Encoder Angle\n(Color by Sample Index)', va='bottom', pad=20)
    # ax.legend() # Legend is less useful for scatter with continuous color

    # Optional: Set radial limits if needed, e.g., based on data range
    # r_min = np.min(r)
    # r_max = np.max(r)
    # ax.set_rlim(bottom=min(0, r_min), top=r_max * 1.1) # Ensure 0 is visible or adapt

    plt.tight_layout()  # Adjust layout to make space for colorbar and title
    plt.savefig(output_file, dpi=300)
    print(f"Color-coded polar vibration plot for {axis_name}-axis saved as {output_file}")
    plt.show()
    plt.close()


def save_axis_fft_results_to_csv(axis_name, freqs, magnitude, peak_freqs, peak_mags, output_dir):
    if len(freqs) == 0: return
    output_file_all = os.path.join(output_dir, f"fft_all_frequencies_{axis_name}.csv")
    output_file_peaks = os.path.join(output_dir, f"fft_peak_frequencies_{axis_name}.csv")
    pd.DataFrame({'Frequency_Hz': freqs, 'Magnitude': magnitude}).to_csv(output_file_all, index=False)
    print(f"Full FFT results for {axis_name}-axis saved to {output_file_all}")
    if len(peak_freqs) > 0:
        pd.DataFrame({'Peak_Frequency_Hz': peak_freqs, 'Peak_Magnitude': peak_mags}).to_csv(output_file_peaks,
                                                                                            index=False)
        print(f"Peak frequencies for {axis_name}-axis saved to {output_file_peaks}")


def main():
    ensure_output_dir()
    input_file = "updated_vibration_data.csv"
    if not os.path.exists(input_file): print(f"Error: {input_file} not found!"); return

    df = load_accelerometer_data(input_file)
    if df.empty: print("Failed to load or process data. Exiting."); return

    print("\n--- DataFrame Info ---")
    df.info()
    print("\n--- NaN counts per column ---")
    print(df.isnull().sum())

    processed_file = os.path.join(OUTPUT_DIR, "processed_data_with_encoder.csv")
    df.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}")

    sample_rate = 0
    if 'Timestamp' in df.columns:
        try:
            timestamps_ms = pd.to_numeric(df['Timestamp'], errors='raise')
            if len(timestamps_ms) > 1:
                avg_diff_ms = np.mean(np.diff(timestamps_ms))
                if avg_diff_ms > 0:
                    sample_rate = 1000.0 / avg_diff_ms
                else:
                    raise ValueError("Timestamp diff not positive.")
            else:
                raise ValueError("Not enough timestamps.")
            print(f"\nInferred sample rate: {sample_rate:.2f} Hz.")
        except Exception as e:
            print(f"Could not infer sample rate from 'Timestamp' ({e}).")

    if sample_rate == 0 and 'Tick' in df.columns:  # Added check for Tick column
        try:
            # Attempt to use Tick if Timestamp failed or wasn't present
            # This assumes ticks are regular and sample_rate can be derived or is known
            # For now, we'll just note if it's present and still rely on manual input if Timestamp fails
            print(
                "Tick column found, but sample rate inference from Tick is not fully implemented. Please provide if Timestamp failed.")
        except Exception as e:
            print(f"Error processing 'Tick' for sample rate: {e}")

    if sample_rate == 0:
        try:
            manual_sample_rate = float(
                input("Could not auto-detect sample rate. Please enter sample rate in Hz (e.g., 2600.0): "))
            if manual_sample_rate <= 0: raise ValueError("Rate must be positive.")
            sample_rate = manual_sample_rate
        except ValueError as e:
            print(f"Invalid rate ({e}). Defaulting to 2600.0 Hz."); sample_rate = 2600.0
    print(f"Using sample rate: {sample_rate:.2f} Hz for FFT analysis.")

    plot_time_domain(df, OUTPUT_DIR, sample_rate_for_time_axis=sample_rate)

    axes_to_analyze = ['X', 'Y', 'Z']
    num_peaks_to_find = 10

    for axis_name in axes_to_analyze:
        print(f"\n--- Processing Analysis for {axis_name}-axis ---")
        signal_data = df[axis_name].fillna(0).values
        if len(signal_data) == 0: print(f"{axis_name}-axis no data. Skipping."); continue

        freqs, magnitude = compute_fft(signal_data, sample_rate)
        if len(freqs) > 0:
            peak_freqs, peak_mags = find_peak_frequencies(freqs, magnitude, num_peaks=num_peaks_to_find)
            print(f"\nTop {len(peak_freqs)} peak frequencies for {axis_name}-axis (sorted by frequency):")
            if len(peak_freqs) > 0:
                for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags)): print(
                    f"{i + 1}. {freq:.2f} Hz (mag: {mag:.6f})")
            else:
                print("No peaks found.")
            plot_fft_results(axis_name, freqs, magnitude, peak_freqs, peak_mags, OUTPUT_DIR)
            save_axis_fft_results_to_csv(axis_name, freqs, magnitude, peak_freqs, peak_mags, OUTPUT_DIR)
        else:
            print(f"FFT computation failed or yielded no frequencies for {axis_name}-axis.")

        print(f"\nGenerating color-coded polar plot for {axis_name}-axis using Encoder data...")
        plot_vibration_polar(axis_name, df, OUTPUT_DIR)

    print(f"\nAll analysis results saved to '{OUTPUT_DIR}/' directory.")
    print("All plots have been displayed.")


if __name__ == "__main__":
    main()