import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import os

# Define the output directory
OUTPUT_DIR = 'fft_results'


def ensure_output_dir():
    """Ensure the output directory exists"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}/")


def load_accelerometer_data(csv_file):
    """Load accelerometer data from CSV file and calculate sample rate."""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    sample_rate = None
    # --- Dynamically calculate sample rate from 'Timestamp' column ---
    if 'Timestamp' in df.columns:
        try:
            # Convert 'Timestamp' to datetime objects
            df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'])

            if len(df) < 2:
                print("Warning: Less than 2 samples, cannot accurately calculate dynamic sample rate from Timestamps.")
            else:
                # Calculate time differences between consecutive samples in seconds
                time_diffs = df['Timestamp_dt'].diff().dt.total_seconds()

                # Calculate average time difference (ignoring the first NaN value)
                avg_time_diff = time_diffs[1:].mean()

                if pd.isna(avg_time_diff) or avg_time_diff <= 0:
                    print(
                        "Warning: Could not determine a valid average time difference from Timestamps. Sample rate calculation failed.")
                else:
                    sample_rate = 1.0 / avg_time_diff
                    print(f"Dynamically calculated average sample rate: {sample_rate:.2f} Hz")
        except Exception as e:
            print(f"Error processing 'Timestamp' column for dynamic sample rate: {e}")
            print("Sample rate calculation failed.")
    else:
        print("Warning: 'Timestamp' column not found. Cannot calculate dynamic sample rate.")
    # --- End of sample rate calculation ---

    # Extract acceleration data from new column names 'X', 'Y', 'Z'
    x_milli = df['X'].values
    y_milli = df['Y'].values
    z_milli = df['Z'].values

    # Convert from milli units to standard units (multiply by 10^-3)
    conversion_factor = 1e-3

    x = x_milli * conversion_factor
    y = y_milli * conversion_factor
    z = z_milli * conversion_factor

    df['X'] = x
    df['Y'] = y
    df['Z'] = z

    print(f"Converted acceleration data from milli units to standard units (factor: {conversion_factor})")

    resultant = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    df['Resultant'] = resultant

    print(f"Loaded {len(df)} samples")
    return df, sample_rate  # Return both df and calculated sample_rate


def compute_fft(signal_data, sample_rate):
    """Compute FFT of a signal"""
    if sample_rate is None or sample_rate <= 0:
        print("Error: Invalid sample rate for FFT computation.")
        return np.array([]), np.array([])

    windowed_signal = signal_data * signal.windows.hann(len(signal_data))
    fft_result = fft(windowed_signal)
    freqs = fftfreq(len(signal_data), 1.0 / sample_rate)
    magnitude = np.abs(fft_result)
    magnitude = magnitude / len(signal_data)

    positive_indices = freqs >= 0
    freqs = freqs[positive_indices]
    magnitude = magnitude[positive_indices]

    return freqs, magnitude


def find_peak_frequencies(freqs, magnitude, num_peaks=10):
    """Find the frequencies with highest magnitude"""
    if len(magnitude) <= 1:
        return np.array([]), np.array([])
    # Ensure we don't try to sort if magnitude is all NaN (can happen if FFT input was bad)
    if np.isnan(magnitude).all():
        return np.array([]), np.array([])

    indices = np.argsort(magnitude[1:])[-num_peaks:] + 1
    indices = sorted(indices)
    peak_freqs = freqs[indices]
    peak_mags = magnitude[indices]

    return peak_freqs, peak_mags


def plot_time_domain(df, sample_rate, output_file=None):  # Added sample_rate argument
    """Plot acceleration data in time domain"""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "time_domain.png")

    plt.figure(figsize=(12, 8))

    # Create time vector using the provided sample rate
    if sample_rate is None or sample_rate <= 0:
        print("Warning: Invalid sample_rate for time domain plot. Assuming 100Hz for timestamps as a fallback.")
        effective_sample_rate = 100.0
    else:
        effective_sample_rate = sample_rate
    timestamps = np.arange(len(df)) / effective_sample_rate

    plt.subplot(4, 1, 1)
    plt.plot(timestamps, df['X'], 'r-', alpha=0.7, label='X-axis')
    plt.title('X-axis Acceleration')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(timestamps, df['Y'], 'g-', alpha=0.7, label='Y-axis')
    plt.title('Y-axis Acceleration')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(timestamps, df['Z'], 'b-', alpha=0.7, label='Z-axis')
    plt.title('Z-axis Acceleration')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(timestamps, df['Resultant'], 'k-', alpha=0.9, label='Resultant')
    plt.title('Resultant Acceleration')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Time domain plot saved as {output_file}")


def plot_fft_results(freqs, magnitude, peak_freqs, peak_mags, output_file=None):
    """Plot FFT results"""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "fft_results.png")
    plt.figure(figsize=(12, 6))
    max_freq = 50

    if len(freqs) == 0:
        print("Warning: No frequency data to plot for FFT results.")
    else:
        mask = freqs <= max_freq
        plt.plot(freqs[mask], magnitude[mask], 'b-', alpha=0.7)

    for f, m in zip(peak_freqs, peak_mags):
        if not pd.isna(m) and f <= max_freq:  # Check for NaN magnitude before plotting
            plt.plot(f, m, 'ro')
            plt.text(f, m * 1.1, f"{f:.2f} Hz", ha='center')

    plt.title('FFT Analysis of Resultant Acceleration')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    if len(freqs) > 0: plt.xlim(left=0)
    if len(magnitude) > 0 and not np.isnan(magnitude).all(): plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"FFT plot saved as {output_file}")


def save_results_to_csv(freqs, magnitude, peak_freqs, peak_mags, output_file=None):
    """Save FFT results to CSV file"""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "results.csv")
    fft_df = pd.DataFrame({'Frequency_Hz': freqs, 'Magnitude': magnitude})
    peaks_df = pd.DataFrame({
        'Peak_Frequency_Hz': peak_freqs,
        'Peak_Magnitude': peak_mags
    })
    fft_df.to_csv(output_file, index=False)
    peaks_file = os.path.join(OUTPUT_DIR, "peak_frequencies.csv")
    peaks_df.to_csv(peaks_file, index=False)
    print(f"FFT results saved to {output_file}")
    print(f"Peak frequencies saved to {peaks_file}")


def main(**kwargs):
    ensure_output_dir()
    input_file = input("Enter you csv file: ")

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        # ... (rest of error message)
        return

    # Load data and get sample_rate
    df, sample_rate = load_accelerometer_data(input_file)  # Modified to get sample_rate

    # --- Debugging (as in your script) ---
    print("--- DataFrame Info ---")
    df.info()
    print("\n--- NaN counts per column ---")
    print(df.isnull().sum())
    print("\n--- First 5 rows of Resultant ---")
    if 'Resultant' in df.columns:
        print(df['Resultant'].head())
    else:
        print("Resultant column not found yet.")
    # --- End Debugging ---

    if df.empty or 'Resultant' not in df.columns:
        print("Failed to load or process data correctly, or Resultant column missing.")
        return

    # --- Handle NaNs in Resultant (Good Practice) ---
    if df['Resultant'].isnull().any():
        print(
            f"\nFound {df['Resultant'].isnull().sum()} NaN values in 'Resultant' column. Dropping rows with these NaNs.")
        df.dropna(subset=['Resultant'], inplace=True)
        print(f"Number of rows after dropping NaNs: {len(df)}")
        if df.empty:
            print("DataFrame is empty after dropping NaNs from Resultant. Cannot proceed.")
            return
    # --- End Handle NaNs ---

    # --- Check if sample rate was successfully calculated ---
    if sample_rate is None:
        print("\nError: Sample rate could not be determined dynamically.")
        print("You might need to check your CSV's 'Timestamp' column or manually set a sample rate.")
        # Option: Fallback to a default or ask user for input
        # For now, we will exit if dynamic calculation fails.
        # sample_rate = 2500 # Example of a fallback
        # print(f"Warning: Using fallback sample rate of {sample_rate} Hz")
        return  # Exit if no valid sample rate

    processed_file = os.path.join(OUTPUT_DIR, "processed_data.csv")
    df.to_csv(processed_file, index=False)
    print(f"Processed data (with resultant) saved to {processed_file}")

    # Plot time domain data using the calculated sample_rate
    plot_time_domain(df, sample_rate)  # Pass sample_rate

    if df['Resultant'].empty:
        print("Resultant column is empty after processing. Cannot compute FFT.")
        return

    freqs, magnitude = compute_fft(df['Resultant'].values, sample_rate)

    if len(freqs) == 0:  # Check if FFT returned empty results
        print("FFT computation resulted in no valid frequencies or magnitudes.")
        return

    peak_freqs, peak_mags = find_peak_frequencies(freqs, magnitude, num_peaks=5)

    print("\nTop 5 peak frequencies:")
    if len(peak_freqs) > 0:
        for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags), 1):
            if pd.isna(mag):
                print(f"{i}. {freq:.2f} Hz (magnitude: nan)")
            else:
                print(f"{i}. {freq:.2f} Hz (magnitude: {mag:.6f})")
    else:
        print("No peaks found.")

    plot_fft_results(freqs, magnitude, peak_freqs, peak_mags)
    save_results_to_csv(freqs, magnitude, peak_freqs, peak_mags)

    print(f"\nAll analysis results have been saved to the '{OUTPUT_DIR}/' directory")


if __name__ == "__main__":
    main()