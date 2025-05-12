import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft  # Added ifft
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
    """Load accelerometer data from CSV file and calculate sample rate"""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    if df.empty:
        print("Error: CSV file is empty.")
        return None, None

    # --- Dynamically calculate sample rate from 'Timestamp' column ---
    if 'Timestamp' not in df.columns:
        print("Error: 'Timestamp' column not found in CSV for sample rate calculation.")
        return df, None  # Return df for potential other uses, but no sample rate

    try:
        # Convert 'Timestamp' to datetime objects
        df['Timestamp_dt'] = pd.to_datetime(df['Timestamp'])

        if len(df) < 2:
            print("Warning: Less than 2 samples, cannot accurately calculate sample rate.")
            sample_rate = None  # Or a default, or raise error
        else:
            # Calculate time differences between consecutive samples in seconds
            time_diffs = df['Timestamp_dt'].diff().dt.total_seconds()

            # Calculate average time difference (ignoring the first NaN value)
            avg_time_diff = time_diffs[1:].mean()

            if pd.isna(avg_time_diff) or avg_time_diff <= 0:
                print("Warning: Could not determine a valid average time difference. Sample rate calculation failed.")
                sample_rate = None
            else:
                sample_rate = 1.0 / avg_time_diff
                print(f"Calculated average sample rate: {sample_rate:.2f} Hz")

    except Exception as e:
        print(f"Error processing 'Timestamp' column for sample rate: {e}")
        sample_rate = None
    # --- End of sample rate calculation ---

    # Extract acceleration data (values are in mili units, 10^-3)
    # --- Updated to use 'X', 'Y', 'Z' column names ---
    if not all(col in df.columns for col in ['X', 'Y', 'Z']):
        print("Error: CSV must contain 'X', 'Y', 'Z' columns for acceleration data.")
        # If sample rate was calculated, still return it, but df might be incomplete for processing
        return df if 'Timestamp_dt' in df else None, sample_rate

    x_mili = df['X'].values
    y_mili = df['Y'].values
    z_mili = df['Z'].values

    # Convert from mili units to standard units (multiply by 10^-3)
    conversion_factor = 1e-3

    x = x_mili * conversion_factor
    y = y_mili * conversion_factor
    z = z_mili * conversion_factor

    # Replace the original values with converted ones (or add as new columns)
    df['Processed_X'] = x  # Storing as new columns to preserve original if needed
    df['Processed_Y'] = y
    df['Processed_Z'] = z

    print(f"Converted acceleration data from mili units to standard units")

    # Compute resultant force using processed data
    resultant = np.sqrt(df['Processed_X'] ** 2 + df['Processed_Y'] ** 2 + df['Processed_Z'] ** 2)

    # Add resultant to DataFrame
    df['Resultant'] = resultant

    print(f"Loaded {len(df)} samples")
    return df, sample_rate


def compute_fft(signal_data, sample_rate):
    """Compute FFT of a signal"""
    N_fft = len(signal_data)
    if N_fft == 0:
        return np.array([]), np.array([]), np.array([]), 0

    # Apply Hanning window to reduce spectral leakage
    windowed_signal = signal_data * signal.windows.hann(N_fft)

    # Compute FFT
    full_fft_coeffs = fft(windowed_signal)

    # Calculate frequency bins
    freqs = fftfreq(N_fft, 1.0 / sample_rate)

    # Calculate magnitude (absolute value)
    magnitude = np.abs(full_fft_coeffs)

    # Normalize (optional, but common)
    # magnitude = magnitude / N_fft # Common normalization for power
    magnitude = magnitude / (N_fft / 2)  # Common normalization for amplitude (except DC and Nyquist)

    # Only take the first half (positive frequencies)
    positive_indices = freqs >= 0
    positive_freqs = freqs[positive_indices]
    positive_magnitude = magnitude[positive_indices]

    return positive_freqs, positive_magnitude, full_fft_coeffs, N_fft


def find_peak_frequencies(freqs, magnitude, num_peaks=5):
    """Find the frequencies with highest magnitude.
       Returns peak frequencies, their magnitudes, and their indices in the input 'freqs' array.
    """
    if len(freqs) <= 1:  # Need at least one point after DC
        return np.array([]), np.array([]), np.array([])

    # Skip DC component (index 0) for peak finding, operate on magnitude[1:]
    # Indices returned by argsort will be for the sliced array (magnitude[1:])
    peak_indices_in_sliced_array = np.argsort(magnitude[1:])[-num_peaks:]

    # Adjust indices to be for the original 'freqs' and 'magnitude' arrays (add 1 back)
    # These indices correspond to the 'freqs' and 'magnitude' arrays passed in.
    actual_peak_indices = peak_indices_in_sliced_array + 1

    # Sort by frequency for better readability (optional, but good for display)
    # Get the frequencies at these peak indices, then sort these frequencies,
    # then get the order of indices that would sort these frequencies.
    sorted_order_of_indices = np.argsort(freqs[actual_peak_indices])

    # Apply this sorted order to actual_peak_indices
    final_sorted_peak_indices = actual_peak_indices[sorted_order_of_indices]

    peak_freqs_output = freqs[final_sorted_peak_indices]
    peak_mags_output = magnitude[final_sorted_peak_indices]

    return peak_freqs_output, peak_mags_output, final_sorted_peak_indices


def plot_time_domain(df, sample_rate, output_file=None):
    """Plot acceleration data in time domain"""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "time_domain.png")

    plt.figure(figsize=(12, 8))

    if sample_rate is None or sample_rate <= 0:
        print("Warning: Valid sample rate not available for time domain plot. Assuming 100Hz for timestamps.")
        sample_rate = 100.0  # Fallback for plotting if sample rate calculation failed

    # Create time vector
    timestamps = np.arange(len(df)) / sample_rate

    plt.subplot(4, 1, 1)
    plt.plot(timestamps, df['Processed_X'], 'r-', alpha=0.7, label='X-axis')
    plt.title('X-axis Acceleration')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(timestamps, df['Processed_Y'], 'g-', alpha=0.7, label='Y-axis')
    plt.title('Y-axis Acceleration')
    plt.ylabel('Acceleration (m/s²)')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(timestamps, df['Processed_Z'], 'b-', alpha=0.7, label='Z-axis')
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


def plot_fft_results(freqs, magnitude, peak_freqs, peak_mags, peak_angles, output_file=None):
    """Plot FFT results"""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "fft_results.png")

    plt.figure(figsize=(12, 6))

    # Limit to 0-50Hz for better visualization of relevant frequencies
    max_freq_plot = 50  # Hz
    mask = (freqs <= max_freq_plot) & (freqs >= 0)  # Ensure positive frequencies

    plt.plot(freqs[mask], magnitude[mask], 'b-', alpha=0.7)

    # Plot peak frequencies
    for f, m, a in zip(peak_freqs, peak_mags, peak_angles):
        if f <= max_freq_plot and f >= 0:
            plt.plot(f, m, 'ro')
            plt.text(f, m * 1.1, f"{f:.2f} Hz\nAngle: {a:.1f}°", ha='center', fontsize=8)

    plt.title('FFT Analysis of Resultant Acceleration')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"FFT plot saved as {output_file}")


def save_results_to_csv(freqs, magnitude, peak_freqs, peak_mags, peak_angles, output_file=None):
    """Save FFT results to CSV file"""
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "fft_summary_results.csv")

    # Create a DataFrame for all FFT results (positive frequencies)
    fft_df = pd.DataFrame({'Frequency_Hz': freqs, 'Magnitude': magnitude})
    # Save all positive frequency components
    all_fft_file = os.path.join(OUTPUT_DIR, "all_fft_components.csv")
    fft_df.to_csv(all_fft_file, index=False)
    print(f"All positive FFT components saved to {all_fft_file}")

    # Create a DataFrame for peak frequencies
    peaks_df = pd.DataFrame({
        'Peak_Frequency_Hz': peak_freqs,
        'Peak_Magnitude': peak_mags,
        'Angle_At_Component_Peak_Deg': peak_angles  # Added angle
    })

    # Write peak results to CSV
    peaks_df.to_csv(output_file, index=False)
    print(f"Peak FFT results saved to {output_file}")


def main(**kwargs):
    ensure_output_dir()

    input_file = input("Please enter the name of your input CSV file (e.g., data.csv): ")

    if not os.path.exists(input_file):
        print(f"Error: Input CSV file '{input_file}' not found!")
        print("Please create this file with your data or specify the correct path.")
        # Example of how the CSV should look:
        print("\nExample CSV format (ensure first line is header):")
        print("Timestamp,Tick,Encoder,X,Y,Z")
        print("2025-05-13T01:34:51.089,8043311,0,-846,-33,601")
        print("2025-05-13T01:34:51.099,8043312,10,-815,-33,531")
        print("...")
        return

    df, sample_rate = load_accelerometer_data(input_file)

    if df is None or df.empty:
        print(f"Failed to load or process data from {input_file}.")
        return
    if 'Resultant' not in df.columns:  # Check if resultant calculation was successful
        print("Error: 'Resultant' column not created. Cannot proceed with FFT.")
        return
    if sample_rate is None or sample_rate <= 0:
        print("Error: Valid sample rate could not be determined. Cannot proceed with FFT.")
        # Optionally, allow user to input sample rate manually here
        # For now, we'll exit.
        # sample_rate = float(input("Please enter the sample rate in Hz: "))
        # if sample_rate <= 0:
        # print("Invalid sample rate entered.")
        return

    processed_file = os.path.join(OUTPUT_DIR, "processed_data_with_resultant.csv")
    df.to_csv(processed_file, index=False)
    print(f"Processed data (with resultant and conversions) saved to {processed_file}")

    plot_time_domain(df, sample_rate)  # Pass sample_rate for correct time axis

    # Compute FFT on resultant
    # Ensure there's data to process
    if df['Resultant'].empty:
        print("Error: Resultant data is empty, cannot compute FFT.")
        return

    resultant_signal = df['Resultant'].values
    positive_freqs, positive_magnitude, full_fft_coeffs, N_fft = compute_fft(resultant_signal, sample_rate)

    if N_fft == 0:
        print("No data to perform FFT on.")
        return

    # Find peak frequencies
    num_peaks_to_find = 5
    peak_freqs, peak_mags, peak_indices_in_pos_freq_array = find_peak_frequencies(
        positive_freqs, positive_magnitude, num_peaks=num_peaks_to_find
    )

    # --- Calculate angles for each peak frequency using IFFT ---
    peak_angles = []
    if 'Encoder' not in df.columns:
        print("Warning: 'Encoder' column not found. Cannot determine angles for peak frequencies.")
        peak_angles = [np.nan] * len(peak_freqs)  # Fill with NaN if encoder data is missing
    elif len(peak_freqs) > 0:
        print("\nCalculating angles for peak frequencies...")
        all_fftfreqs = fftfreq(N_fft, 1.0 / sample_rate)  # Full frequency spectrum for indexing

        for i, specific_peak_freq in enumerate(peak_freqs):
            # Find the index k in the full_fft_coeffs that corresponds to this peak_freqs[i]
            # peak_indices_in_pos_freq_array contains indices relative to positive_freqs.
            # These are the correct k values for the positive side of full_fft_coeffs.
            k = peak_indices_in_pos_freq_array[i]

            single_component_fft = np.zeros_like(full_fft_coeffs, dtype=complex)
            single_component_fft[k] = full_fft_coeffs[k]

            # Add conjugate for negative frequency if not DC (k=0) or Nyquist (k=N_fft/2 if N_fft is even)
            if k != 0 and k * 2 != N_fft:  # Check k*2 != N_fft for Nyquist
                single_component_fft[N_fft - k] = full_fft_coeffs[N_fft - k]

            # Inverse FFT to get time domain signal of this single component
            time_signal_of_component = np.real(ifft(single_component_fft))

            # Find index of max amplitude in this component's time signal
            if len(time_signal_of_component) > 0:
                max_energy_idx = np.argmax(np.abs(time_signal_of_component))
                angle_at_peak = df['Encoder'].iloc[max_energy_idx]
                peak_angles.append(angle_at_peak)
            else:
                peak_angles.append(np.nan)  # Should not happen if N_fft > 0
        print("Angle calculation complete.")
    else:  # No peaks found
        peak_angles = []

    print("\nTop peak frequencies:")
    for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags), 1):
        angle_str = f", angle: {peak_angles[i - 1]:.2f}°" if peak_angles and not pd.isna(
            peak_angles[i - 1]) else ", angle: N/A"
        print(f"{i}. {freq:.2f} Hz (magnitude: {mag:.4f}{angle_str})")

    plot_fft_results(positive_freqs, positive_magnitude, peak_freqs, peak_mags, peak_angles)
    save_results_to_csv(positive_freqs, positive_magnitude, peak_freqs, peak_mags, peak_angles)

    print(f"\nAll analysis results have been saved to the '{OUTPUT_DIR}/' directory")


if __name__ == "__main__":
    # To run this script, save it as fft.py, create a CSV file
    # (e.g., "my_vibration_data.csv") with your data,
    # and then run from the command line:
    # python fft.py
    # Or, if you want to specify the file:
    # main(input_csv="my_vibration_data.csv")

    # For testing, you might create a dummy CSV named "your_data.csv"
    # or pass the actual filename if your data is ready.
    # Example:
    # main(input_csv="path_to_your/thirtyEightHz.csv")
    main()