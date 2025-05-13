import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import os

"""
THIS ONE IS FOR MY BEST FRIEND ALPEREN
WHO IS CRYING A LOT THESE DAYS.
"""

# Define the output directory
OUTPUT_DIR = 'fft_results_xyz'


def ensure_output_dir():
    """Ensure the output directory exists"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}/")


def load_accelerometer_data(csv_file):
    """Load accelerometer data from CSV file and convert to standard units"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    if not all(col in df.columns for col in ['X', 'Y', 'Z']):
        print("Error: CSV file must contain 'X', 'Y', and 'Z' columns.")
        return pd.DataFrame()  # Return empty DataFrame on error

    # Extract acceleration data from columns 'X', 'Y', 'Z'
    x_milli = df['X'].values
    y_milli = df['Y'].values
    z_milli = df['Z'].values

    # Convert from milli units to standard units (e.g., milli-g to g, or milli-m/s^2 to m/s^2)
    conversion_factor = 1e-3 * 9.81 # Assuming original units were milli-X (e.g. milli-g or milli-m/s^2)

    x = x_milli * conversion_factor
    y = y_milli * conversion_factor
    z = z_milli * conversion_factor

    # Replace the original values in the columns with converted ones
    df['X'] = x
    df['Y'] = y
    df['Z'] = z

    print(f"Converted X, Y, Z acceleration data from milli units to standard units (factor: {conversion_factor})")
    print(f"Loaded {len(df)} samples")
    return df


def compute_fft(signal_data, sample_rate):
    """Compute FFT of a signal"""
    if len(signal_data) == 0:
        print("Warning: Empty signal data provided to compute_fft.")
        return np.array([]), np.array([])

    # Apply Hanning window to reduce spectral leakage
    windowed_signal = signal_data * signal.windows.hann(len(signal_data))

    # Compute FFT
    fft_result = fft(windowed_signal)

    # Calculate frequency bins
    freqs = fftfreq(len(signal_data), 1.0 / sample_rate)

    # Calculate magnitude (absolute value)
    magnitude = np.abs(fft_result)

    # Normalize by the number of points for correct amplitude scaling
    magnitude = magnitude / len(signal_data)

    # Only take the first half (positive frequencies)
    positive_indices = freqs >= 0
    freqs = freqs[positive_indices]
    magnitude = magnitude[positive_indices]

    # Double the magnitude for all but DC (0 Hz) and Nyquist frequency components
    # to account for energy in negative frequencies (if signal is real)
    if len(magnitude) > 1:  # Avoid error on empty or single point magnitude
        magnitude[1:] = magnitude[1:] * 2

    return freqs, magnitude


def find_peak_frequencies(freqs, magnitude, num_peaks=5):
    """Find the frequencies with highest magnitude"""
    if len(magnitude) <= 1:  # Handle cases with very short signals or no positive frequencies
        return np.array([]), np.array([])

    search_freqs = freqs
    search_magnitude = magnitude
    offset = 0
    # If freqs[0] is 0 (DC component), search peaks in magnitude[1:]
    if freqs[0] == 0 and len(freqs) > 1:
        search_freqs = freqs[1:]
        search_magnitude = magnitude[1:]
        offset = 1  # to adjust indices back
    elif freqs[0] == 0 and len(freqs) == 1:  # Only DC component exists
        return np.array([]), np.array([])

    if len(search_magnitude) == 0:
        return np.array([]), np.array([])

    actual_num_peaks = min(num_peaks, len(search_magnitude))
    if actual_num_peaks == 0:
        return np.array([]), np.array([])

    indices = np.argsort(search_magnitude)[-actual_num_peaks:] + offset

    sorted_indices_by_freq = sorted(indices, key=lambda i: freqs[i])

    peak_freqs = freqs[sorted_indices_by_freq]
    peak_mags = magnitude[sorted_indices_by_freq]

    return peak_freqs, peak_mags


def plot_time_domain(df, output_dir, sample_rate_for_time_axis):
    """Plot acceleration data in time domain for X, Y, Z axes"""
    output_file = os.path.join(output_dir, "time_domain_XYZ.png")

    plt.figure(figsize=(12, 9))

    if 'Timestamp' in df.columns:
        try:
            numeric_timestamps = pd.to_numeric(df['Timestamp'], errors='coerce')
            if not numeric_timestamps.isnull().all():
                timestamps = (numeric_timestamps - numeric_timestamps.iloc[0]) / 1000.0
                print("Using 'Timestamp' column for time axis (converted to seconds).")
            else:
                raise ValueError("Timestamp column not suitable for numeric conversion or all NaNs.")
        except Exception as e:
            print(f"Could not use 'Timestamp' for time axis ({e}), falling back.")
            timestamps = np.arange(len(df)) / sample_rate_for_time_axis
            print(f"Using index-based time axis with sample rate: {sample_rate_for_time_axis} Hz.")
    elif 'Tick' in df.columns:
        timestamps = df['Tick'].values / sample_rate_for_time_axis
        print(f"Using 'Tick' column for time axis with sample rate: {sample_rate_for_time_axis} Hz.")
    else:
        timestamps = np.arange(len(df)) / sample_rate_for_time_axis
        print(f"Using index-based time axis with sample rate: {sample_rate_for_time_axis} Hz.")

    plt.subplot(3, 1, 1)
    plt.plot(timestamps, df['X'], 'r-', alpha=0.7, label='X-axis')
    plt.title('X-axis Acceleration')
    plt.ylabel('Acceleration (units)')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(timestamps, df['Y'], 'g-', alpha=0.7, label='Y-axis')
    plt.title('Y-axis Acceleration')
    plt.ylabel('Acceleration (units)')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(timestamps, df['Z'], 'b-', alpha=0.7, label='Z-axis')
    plt.title('Z-axis Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (units)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Time domain plot saved as {output_file}")
    plt.show()
    plt.close()


def plot_fft_results(axis_name, freqs, magnitude, peak_freqs, peak_mags, output_dir):
    """Plot FFT results for a given axis (full Nyquist range) and show it"""
    output_file = os.path.join(output_dir, f"fft_results_{axis_name}.png")

    plt.figure(figsize=(12, 6))

    if len(freqs) == 0:
        print(f"Warning: No frequency data to plot for FFT results ({axis_name}-axis).")
    else:
        # Plot all available positive frequencies
        plt.plot(freqs, magnitude, 'b-', alpha=0.7)

    # Plot peak frequencies - they will be plotted regardless of any previous max_freq_plot
    for f, m in zip(peak_freqs, peak_mags):
        plt.plot(f, m, 'ro')
        plt.text(f, m * 1.1, f"{f:.2f} Hz", ha='center')

    plt.title(f'FFT Analysis of {axis_name}-axis Acceleration (Full Spectrum)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    # Set x-axis to start at 0 and go up to the highest frequency available (Nyquist)
    if len(freqs) > 0:
        plt.xlim(left=0, right=freqs[-1] if len(freqs) > 0 else None)
    if len(magnitude) > 0:
        plt.ylim(bottom=0)  # Ensure y-axis starts at 0

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"FFT plot for {axis_name}-axis saved as {output_file}")
    plt.show()
    plt.close()


def save_axis_fft_results_to_csv(axis_name, freqs, magnitude, peak_freqs, peak_mags, output_dir):
    """Save FFT results for a given axis to CSV files"""
    if len(freqs) == 0:
        print(f"No FFT data to save for {axis_name}-axis.")
        return

    output_file_all = os.path.join(output_dir, f"fft_all_frequencies_{axis_name}.csv")
    output_file_peaks = os.path.join(output_dir, f"fft_peak_frequencies_{axis_name}.csv")

    fft_df = pd.DataFrame({'Frequency_Hz': freqs, 'Magnitude': magnitude})
    fft_df.to_csv(output_file_all, index=False)
    print(f"Full FFT results for {axis_name}-axis saved to {output_file_all}")

    if len(peak_freqs) > 0:
        peaks_df = pd.DataFrame({'Peak_Frequency_Hz': peak_freqs, 'Peak_Magnitude': peak_mags})
        peaks_df.to_csv(output_file_peaks, index=False)
        print(f"Peak frequencies for {axis_name}-axis saved to {output_file_peaks}")
    else:
        print(f"No peak frequencies to save for {axis_name}-axis.")


def main():
    ensure_output_dir()

    input_file = input("Enter your CSV file path: ")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    df = load_accelerometer_data(input_file)

    if df.empty:
        print("Failed to load or process data. Exiting.")
        return

    print("\n--- DataFrame Info ---")
    df.info()
    print("\n--- NaN counts per column ---")
    print(df.isnull().sum())
    for axis_col in ['X', 'Y', 'Z']:
        if axis_col in df.columns:
            print(f"\n--- First 5 rows of {axis_col}-axis (converted) ---")
            print(df[axis_col].head())
        else:
            print(f"{axis_col} column not found after loading.")
            return

    processed_file = os.path.join(OUTPUT_DIR, "processed_data_XYZ.csv")
    df.to_csv(processed_file, index=False)
    print(f"Processed data (X,Y,Z converted) saved to {processed_file}")

    sample_rate = 0
    if 'Timestamp' in df.columns:
        try:
            timestamps_ms = pd.to_numeric(df['Timestamp'], errors='raise')
            if len(timestamps_ms) > 1:
                avg_diff_ms = np.mean(np.diff(timestamps_ms))
                if avg_diff_ms > 0:
                    sample_rate = 1000.0 / avg_diff_ms
                    print(f"\nInferred sample rate from 'Timestamp' column: {sample_rate:.2f} Hz.")
                else:
                    raise ValueError("Timestamp differences are not positive.")
            else:
                raise ValueError("Not enough timestamps to calculate rate.")
        except Exception as e:
            print(f"Could not infer sample rate from 'Timestamp' ({e}).")

    if sample_rate == 0 and 'Tick' in df.columns:
        try:
            print("Could not reliably infer sample rate from 'Tick' without more information (e.g., total duration).")
        except Exception as e:
            print(f"Error processing 'Tick' for sample rate: {e}")

    if sample_rate == 0:
        try:
            manual_sample_rate = float(
                input("Could not auto-detect sample rate. Please enter sample rate in Hz (e.g., 2600.0): "))
            if manual_sample_rate <= 0:
                raise ValueError("Sample rate must be positive.")
            sample_rate = manual_sample_rate
        except ValueError as e:
            print(f"Invalid sample rate entered ({e}). Defaulting to 2600.0 Hz.")
            sample_rate = 2600.0

    print(f"Using sample rate: {sample_rate:.2f} Hz for FFT analysis.")

    plot_time_domain(df, OUTPUT_DIR, sample_rate_for_time_axis=sample_rate)

    axes_to_analyze = ['X', 'Y', 'Z']
    num_peaks_to_find = 10

    for axis_name in axes_to_analyze:
        print(f"\n--- Processing FFT for {axis_name}-axis ---")
        signal_data = df[axis_name].values

        if df[axis_name].isnull().any():
            print(f"Warning: {axis_name}-axis contains NaN values. Filling with 0 for FFT.")
            signal_data = df[axis_name].fillna(0).values

        if len(signal_data) == 0:
            print(f"{axis_name}-axis has no data. Skipping FFT.")
            continue

        freqs, magnitude = compute_fft(signal_data, sample_rate)

        if len(freqs) == 0:
            print(f"FFT computation failed or yielded no frequencies for {axis_name}-axis.")
            continue

        peak_freqs, peak_mags = find_peak_frequencies(freqs, magnitude, num_peaks=num_peaks_to_find)

        print(f"\nTop {len(peak_freqs)} peak frequencies for {axis_name}-axis (sorted by frequency):")
        if len(peak_freqs) > 0:
            for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags)):
                print(f"{i + 1}. {freq:.2f} Hz (magnitude: {mag:.6f})")
        else:
            print("No peaks found.")

        plot_fft_results(axis_name, freqs, magnitude, peak_freqs, peak_mags, OUTPUT_DIR)
        save_axis_fft_results_to_csv(axis_name, freqs, magnitude, peak_freqs, peak_mags, OUTPUT_DIR)

    print(f"\nAll analysis results have been saved to the '{OUTPUT_DIR}/' directory.")
    print("All plots have been displayed.")


if __name__ == "__main__":
    main()