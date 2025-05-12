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
    """Load accelerometer data from CSV file"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Extract acceleration data (values are in mili units, 10^-3)
    x_mili = df['AxisX'].values
    y_mili = df['AxisY'].values
    z_mili = df['AxisZ'].values

    # Convert from mili units to standard units (multiply by 10^-3)
    conversion_factor = 1e-3

    x = x_mili * conversion_factor
    y = y_mili * conversion_factor
    z = z_mili * conversion_factor

    # Replace the original values with converted ones
    df['AxisX'] = x
    df['AxisY'] = y
    df['AxisZ'] = z

    print(f"Converted acceleration data from mili units to standard units")

    # Compute resultant force
    resultant = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Add resultant to DataFrame
    df['Resultant'] = resultant

    print(f"Loaded {len(df)} samples")
    return df


def compute_fft(signal_data, sample_rate):
    """Compute FFT of a signal"""
    # Apply Hanning window to reduce spectral leakage
    windowed_signal = signal_data * signal.windows.hann(len(signal_data))

    # Compute FFT
    fft_result = fft(windowed_signal)

    # Calculate frequency bins
    freqs = fftfreq(len(signal_data), 1.0 / sample_rate)

    # Calculate magnitude (absolute value)
    magnitude = np.abs(fft_result)

    # Normalize
    magnitude = magnitude / len(signal_data)

    # Only take the first half (positive frequencies)
    positive_indices = freqs >= 0
    freqs = freqs[positive_indices]
    magnitude = magnitude[positive_indices]

    return freqs, magnitude


def find_peak_frequencies(freqs, magnitude, num_peaks=5):
    """Find the frequencies with highest magnitude"""
    # Skip DC component (index 0)
    indices = np.argsort(magnitude[1:])[-num_peaks:] + 1

    # Sort by frequency for better readability
    indices = sorted(indices)

    peak_freqs = freqs[indices]
    peak_mags = magnitude[indices]

    return peak_freqs, peak_mags


def plot_time_domain(df, output_file=None):
    """Plot acceleration data in time domain"""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "time_domain.png")

    plt.figure(figsize=(12, 8))

    # Create time vector (assuming constant sample rate)
    timestamps = np.arange(len(df)) / 100.0  # Assuming 100Hz sample rate

    plt.subplot(4, 1, 1)
    plt.plot(timestamps, df['AxisX'], 'r-', alpha=0.7, label='X-axis')
    plt.title('X-axis Acceleration')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(timestamps, df['AxisY'], 'g-', alpha=0.7, label='Y-axis')
    plt.title('Y-axis Acceleration')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(timestamps, df['AxisZ'], 'b-', alpha=0.7, label='Z-axis')
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

    # Limit to 0-50Hz for better visualization of relevant frequencies
    max_freq = 50  # Hz
    mask = freqs <= max_freq

    plt.plot(freqs[mask], magnitude[mask], 'b-', alpha=0.7)

    # Plot peak frequencies
    for f, m in zip(peak_freqs, peak_mags):
        if f <= max_freq:
            plt.plot(f, m, 'ro')
            plt.text(f, m * 1.1, f"{f:.2f} Hz", ha='center')

    plt.title('FFT Analysis of Resultant Acceleration')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"FFT plot saved as {output_file}")


def save_results_to_csv(freqs, magnitude, peak_freqs, peak_mags, output_file=None):
    """Save FFT results to CSV file"""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "results.csv")

    # Create a DataFrame for all FFT results
    fft_df = pd.DataFrame({'Frequency_Hz': freqs, 'Magnitude': magnitude})

    # Create a DataFrame for peak frequencies
    peaks_df = pd.DataFrame({
        'Peak_Frequency_Hz': peak_freqs,
        'Peak_Magnitude': peak_mags
    })

    # Write both to CSV
    fft_df.to_csv(output_file, index=False)

    # Also write the peaks to a separate file for easier viewing
    peaks_file = os.path.join(OUTPUT_DIR, "peak_frequencies.csv")
    peaks_df.to_csv(peaks_file, index=False)

    print(f"FFT results saved to {output_file}")
    print(f"Peak frequencies saved to {peaks_file}")


def main(**kwargs):
    # Ensure output directory exists
    ensure_output_dir()

    # Input file (change this to your CSV file)
    input_file = "thirtyEightHz.csv"

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    # Load data
    df = load_accelerometer_data(input_file)

    # Save processed data with resultant to output directory
    processed_file = os.path.join(OUTPUT_DIR, "processed_data.csv")
    df.to_csv(processed_file, index=False)
    print(f"Processed data (with resultant) saved to {processed_file}")

    # Sample rate (assumed from DATA_RATE_HZ in your code)
    sample_rate = 100.0  # Hz

    # Plot time domain data
    plot_time_domain(df)

    # Compute FFT on resultant
    freqs, magnitude = compute_fft(df['Resultant'].values, sample_rate)

    # Find peak frequencies
    peak_freqs, peak_mags = find_peak_frequencies(freqs, magnitude, num_peaks=5)

    # Print peak frequencies
    print("\nTop 5 peak frequencies:")
    for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags), 1):
        print(f"{i}. {freq:.2f} Hz (magnitude: {mag:.6f})")

    # Plot FFT results
    plot_fft_results(freqs, magnitude, peak_freqs, peak_mags)

    # Save results
    save_results_to_csv(freqs, magnitude, peak_freqs, peak_mags)

    print(f"\nAll analysis results have been saved to the '{OUTPUT_DIR}/' directory")


if __name__ == "__main__":
    main()