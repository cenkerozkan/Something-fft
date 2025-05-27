import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import os

"""
THIS ONE IS CALCULATING FFT WITH RESULTANT ACCELERATION.
SAMPLING RATE IS STATIC (HARDCODED)
"""

# Define the output directory
OUTPUT_DIR = 'fft_results'  # Output directory for this version


def ensure_output_dir():
    """Ensure the output directory exists"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}/")


def load_accelerometer_data(csv_file):
    """Load accelerometer data from CSV file"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    if not all(col in df.columns for col in ['X', 'Y', 'Z']):
        print("Error: CSV file must contain 'X', 'Y', and 'Z' columns.")
        return pd.DataFrame()

    # Extract acceleration data from new column names 'X', 'Y', 'Z'
    x_milli = df['X'].values
    y_milli = df['Y'].values
    z_milli = df['Z'].values

    # Convert from milli units to standard units (multiply by 10^-3)
    conversion_factor = 1e-3

    x = x_milli * conversion_factor
    y = y_milli * conversion_factor
    z = z_milli * conversion_factor

    # Replace the original values in the new columns with converted ones
    df['X'] = x
    df['Y'] = y
    df['Z'] = z

    print(f"Converted acceleration data from milli units to standard units (factor: {conversion_factor})")

    # Compute resultant force using the converted X, Y, Z values
    resultant = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Add resultant to DataFrame
    df['Resultant'] = resultant

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

    # Normalize by the number of points
    magnitude = magnitude / len(signal_data)

    # Only take the first half (positive frequencies)
    positive_indices = freqs >= 0
    freqs = freqs[positive_indices]
    magnitude = magnitude[positive_indices]

    # Double the magnitude for all but DC (0 Hz) and Nyquist frequency components
    # to account for energy in negative frequencies (if signal is real)
    if len(magnitude) > 1:
        magnitude[1:] = magnitude[1:] * 2

    return freqs, magnitude


def find_peak_frequencies(freqs, magnitude, num_peaks=5):
    """Find the frequencies with highest magnitude"""
    if len(magnitude) <= 1:
        return np.array([]), np.array([])

    search_freqs = freqs
    search_magnitude = magnitude
    offset = 0
    if freqs[0] == 0 and len(freqs) > 1:  # If DC is present and there are other frequencies
        search_freqs = freqs[1:]
        search_magnitude = magnitude[1:]
        offset = 1
    elif freqs[0] == 0 and len(freqs) == 1:  # Only DC component
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


def plot_time_domain(df, sample_rate_for_time_axis, output_file=None):
    """Plot acceleration data in time domain"""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "time_domain_resultant.png")

    plt.figure(figsize=(12, 8))

    # Create time vector
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

    plt.subplot(4, 1, 1)
    plt.plot(timestamps, df['X'], 'r-', alpha=0.7, label='X-axis')
    plt.title('X-axis Acceleration')
    plt.ylabel('Acceleration (units)')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(timestamps, df['Y'], 'g-', alpha=0.7, label='Y-axis')
    plt.title('Y-axis Acceleration')
    plt.ylabel('Acceleration (units)')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(timestamps, df['Z'], 'b-', alpha=0.7, label='Z-axis')
    plt.title('Z-axis Acceleration')
    plt.ylabel('Acceleration (units)')
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(timestamps, df['Resultant'], 'k-', alpha=0.9, label='Resultant')
    plt.title('Resultant Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (units)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Time domain plot saved as {output_file}")
    plt.show()
    plt.close()


def plot_fft_results(freqs, magnitude, peak_freqs, peak_mags, output_file=None):
    """Plot FFT results (Full Nyquist Range)"""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "fft_results_resultant.png")

    plt.figure(figsize=(12, 6))

    if len(freqs) == 0:
        print("Warning: No frequency data to plot for FFT results.")
    else:
        # Plot all available positive frequencies
        plt.plot(freqs, magnitude, 'b-', alpha=0.7)

    # Plot peak frequencies
    for f, m in zip(peak_freqs, peak_mags):
        # Peaks will be plotted if they exist, regardless of previous display limits
        plt.plot(f, m, 'ro')
        plt.text(f, m * 1.1, f"{f:.2f} Hz", ha='center')

    plt.title('FFT Analysis of Resultant Acceleration (Full Spectrum)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    # Ensure axes start at 0 or above, and x-axis extends to Nyquist
    if len(freqs) > 0:
        plt.xlim(left=0, right=freqs[-1] if len(freqs) > 0 else None)
    if len(magnitude) > 0:
        plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"FFT plot for Resultant saved as {output_file}")
    plt.show()
    plt.close()


def save_results_to_csv(freqs, magnitude, peak_freqs, peak_mags, output_file=None):
    """Save FFT results to CSV file"""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "fft_all_frequencies_resultant.csv")

    peaks_csv_file = os.path.join(OUTPUT_DIR, "fft_peak_frequencies_resultant.csv")

    if len(freqs) > 0:
        fft_df = pd.DataFrame({'Frequency_Hz': freqs, 'Magnitude': magnitude})
        fft_df.to_csv(output_file, index=False)
        print(f"Full FFT results for Resultant saved to {output_file}")
    else:
        print("No full FFT data to save for Resultant.")

    if len(peak_freqs) > 0:
        peaks_df = pd.DataFrame({
            'Peak_Frequency_Hz': peak_freqs,
            'Peak_Magnitude': peak_mags
        })
        peaks_df.to_csv(peaks_csv_file, index=False)
        print(f"Peak frequencies for Resultant saved to {peaks_csv_file}")
    else:
        print("No peak frequencies to save for Resultant.")


def main():
    ensure_output_dir()

    input_file = input("Enter your CSV file path: ")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        print("Please ensure your CSV file is correctly named and in the path,")
        print("or modify the 'input_file' variable in the script.")
        print("The CSV should have columns: Timestamp,Tick,Encoder,X,Y,Z")
        return

    df = load_accelerometer_data(input_file)

    if df.empty:
        print("DataFrame is empty after loading. Exiting.")
        return
    if 'Resultant' not in df.columns:
        print("Critical 'Resultant' column is missing after loading. Exiting.")
        return

    print("\n--- DataFrame Info ---")
    df.info()
    print("\n--- NaN counts per column ---")
    print(df.isnull().sum())
    print("\n--- First 5 rows of Resultant ---")
    print(df['Resultant'].head())

    # Save processed data with resultant to output directory
    processed_file = os.path.join(OUTPUT_DIR, "processed_data_resultant.csv")
    df.to_csv(processed_file, index=False)
    print(f"Processed data (with resultant) saved to {processed_file}")

    # --- Sample Rate Determination ---
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
            # This part is tricky without knowing the exact nature of 'Tick'
            # For simplicity, we'll assume if Timestamp fails, user must provide or use default.
            print("Could not reliably infer sample rate from 'Tick' without more information.")
        except Exception as e:
            print(f"Error processing 'Tick' for sample rate: {e}")

    if sample_rate == 0:  # Fallback if not determined
        try:
            manual_sample_rate = float(
                input("Could not auto-detect sample rate. Please enter sample rate in Hz (e.g., 2600.0): "))
            if manual_sample_rate <= 0:
                raise ValueError("Sample rate must be positive.")
            sample_rate = manual_sample_rate
        except ValueError as e:
            print(f"Invalid sample rate entered ({e}). Defaulting to 2600.0 Hz.")
            sample_rate = 2600.0  # Default fallback

    print(f"Using sample rate: {sample_rate:.2f} Hz for FFT analysis.")
    # --- End Sample Rate Determination ---

    plot_time_domain(df, sample_rate_for_time_axis=sample_rate)

    if df['Resultant'].empty:
        print("Resultant column is empty. Cannot compute FFT.")
        return
    if df['Resultant'].isnull().all():
        print("Resultant column contains only NaN values. Cannot compute FFT.")
        return

    signal_to_fft = df['Resultant'].fillna(0).values  # Handle potential NaNs by filling with 0

    freqs, magnitude = compute_fft(signal_to_fft, sample_rate)

    if len(freqs) == 0:
        print("FFT computation failed or yielded no frequencies for Resultant.")
        return

    peak_freqs, peak_mags = find_peak_frequencies(freqs, magnitude, num_peaks=10)

    print("\nTop peak frequencies for Resultant (sorted by frequency):")
    if len(peak_freqs) > 0:
        for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags), 1):
            print(f"{i}. {freq:.2f} Hz (magnitude: {mag:.6f})")
    else:
        print("No peaks found.")

    plot_fft_results(freqs, magnitude, peak_freqs, peak_mags)
    save_results_to_csv(freqs, magnitude, peak_freqs, peak_mags)

    print(f"\nAll analysis results have been saved to the '{OUTPUT_DIR}/' directory")
    print("All plots have been displayed.")


if __name__ == "__main__":
    main()