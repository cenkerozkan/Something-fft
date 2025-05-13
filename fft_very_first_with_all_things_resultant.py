import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import os

# Define the output directory
OUTPUT_DIR = 'balancing_analysis_resultant'


def ensure_output_dir():
    """Ensure the output directory exists"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}/")


def load_accelerometer_data(csv_file):
    """Load accelerometer data, convert, and calculate Resultant."""
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    required_cols = ['X', 'Y', 'Z', 'Encoder']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file must contain {', '.join(required_cols)} columns.")
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f"Missing columns: {', '.join(missing_cols)}")
        return pd.DataFrame()

    # Convert X, Y, Z to standard units
    conversion_factor = 1e-3
    df['X_conv'] = df['X'] * conversion_factor
    df['Y_conv'] = df['Y'] * conversion_factor
    df['Z_conv'] = df['Z'] * conversion_factor
    print(f"Converted X, Y, Z acceleration data to standard units (factor: {conversion_factor})")

    # Calculate Resultant using converted values
    df['Resultant'] = np.sqrt(df['X_conv'] ** 2 + df['Y_conv'] ** 2 + df['Z_conv'] ** 2)
    print("Calculated 'Resultant' acceleration.")

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


def plot_time_domain_all(df, output_dir, sample_rate_for_time_axis):
    """Plots X, Y, Z, and Resultant acceleration in time domain."""
    output_file = os.path.join(output_dir, "time_domain_XYZ_Resultant.png")
    plt.figure(figsize=(12, 10))  # Adjusted for 4 subplots
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

    axes_to_plot = {'X_conv': 'X-axis (converted)', 'Y_conv': 'Y-axis (converted)', 'Z_conv': 'Z-axis (converted)',
                    'Resultant': 'Resultant'}
    colors = ['r', 'g', 'b', 'k']

    for i, (col_name, title_name) in enumerate(axes_to_plot.items()):
        plt.subplot(4, 1, i + 1)
        plt.plot(timestamps, df[col_name].fillna(0), label=title_name, color=colors[i], alpha=0.7)
        plt.title(title_name + ' Acceleration')
        plt.ylabel('Acceleration (units)')
        plt.grid(True)
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Time domain plot (X,Y,Z,Resultant) saved as {output_file}")
    plt.show()
    plt.close()


def plot_fft_results(axis_name, freqs, magnitude, peak_freqs, peak_mags, output_dir):
    # (This function remains the same as before)
    output_file = os.path.join(output_dir, f"fft_results_{axis_name}.png")
    plt.figure(figsize=(12, 6))
    if len(freqs) == 0:
        print(f"Warning: No FFT data for {axis_name}.")
    else:
        plt.plot(freqs, magnitude, 'b-', alpha=0.7)
    for f, m in zip(peak_freqs, peak_mags): plt.plot(f, m, 'ro'); plt.text(f, m * 1.1, f"{f:.2f} Hz", ha='center')
    plt.title(f'FFT Analysis of {axis_name} Acceleration (Full Spectrum)')
    plt.xlabel('Frequency (Hz)');
    plt.ylabel('Magnitude');
    plt.grid(True)
    if len(freqs) > 0: plt.xlim(left=0, right=freqs[-1] if len(freqs) > 0 else None)
    if len(magnitude) > 0: plt.ylim(bottom=0)
    plt.tight_layout();
    plt.savefig(output_file, dpi=300);
    print(f"FFT plot for {axis_name} saved as {output_file}");
    plt.show();
    plt.close()


def plot_vibration_polar(signal_name, df_signal_values, df_encoder_values, output_dir):
    """
    Plots vibration for a given signal against encoder angle on a polar chart,
    colored by sample index, and marks the angle of maximum vibration.
    Returns the angle (in degrees) of maximum observed vibration.
    """
    if df_encoder_values.isnull().all():
        print(f"Warning: 'Encoder' column all NaN for {signal_name}. Skipping polar plot.")
        return None

    # Create a temporary DataFrame for clean processing
    temp_df = pd.DataFrame({
        'signal': df_signal_values,
        'Encoder': df_encoder_values,
        'original_index': df_signal_values.index  # Keep original index for color
    })
    clean_df = temp_df.dropna(subset=['signal', 'Encoder'])

    if clean_df.empty:
        print(f"No valid data points for {signal_name} and Encoder after dropping NaNs. Skipping polar plot.")
        return None

    output_file = os.path.join(output_dir, f"polar_vibration_color_{signal_name}.png")
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})

    theta = np.deg2rad(clean_df['Encoder'].values)
    r = clean_df['signal'].values
    colors = clean_df['original_index']  # Use original index for consistent coloring

    scatter = ax.scatter(theta, r, c=colors, cmap='viridis', alpha=0.6, s=10)
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.1)
    cbar.set_label('Original Sample Index (Time Progression)')

    angle_max_vib_deg = None
    if len(r) > 0:
        idx_max_r = np.argmax(r)
        theta_max_r = theta[idx_max_r]
        val_max_r = r[idx_max_r]
        angle_max_vib_deg = clean_df['Encoder'].iloc[idx_max_r]

        ax.plot(theta_max_r, val_max_r, 'X', color='red', markersize=12,
                label=f'Max Vib: {val_max_r:.2f} @ {angle_max_vib_deg:.1f}°')
        print(f"Max vibration for {signal_name}: {val_max_r:.3f} at Encoder Angle: {angle_max_vib_deg:.1f}°")

    ax.set_title(f'{signal_name} Vibration vs. Encoder Angle\n(Color by Sample Index)', va='bottom', pad=20)
    if angle_max_vib_deg is not None:
        ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Color-coded polar vibration plot for {signal_name} saved as {output_file}")
    plt.show()
    plt.close()

    return angle_max_vib_deg


def save_axis_fft_results_to_csv(axis_name, freqs, magnitude, peak_freqs, peak_mags, output_dir):
    # (This function remains the same as before)
    if len(freqs) == 0: return
    output_file_all = os.path.join(output_dir, f"fft_all_frequencies_{axis_name}.csv")
    output_file_peaks = os.path.join(output_dir, f"fft_peak_frequencies_{axis_name}.csv")
    pd.DataFrame({'Frequency_Hz': freqs, 'Magnitude': magnitude}).to_csv(output_file_all, index=False)
    print(f"Full FFT results for {axis_name} saved to {output_file_all}")
    if len(peak_freqs) > 0:
        pd.DataFrame({'Peak_Frequency_Hz': peak_freqs, 'Peak_Magnitude': peak_mags}).to_csv(output_file_peaks,
                                                                                            index=False)
        print(f"Peak frequencies for {axis_name} saved to {output_file_peaks}")


def main():
    ensure_output_dir()
    input_file = input("Enter your CSV file path: ")
    if not os.path.exists(input_file): print(f"Error: {input_file} not found!"); return

    df = load_accelerometer_data(input_file)
    if df.empty: print("Failed to load or process data. Exiting."); return
    if 'Resultant' not in df.columns: print("'Resultant' column missing. Exiting."); return
    if 'Encoder' not in df.columns: print("'Encoder' column missing. Exiting."); return

    print("\n--- DataFrame Info ---")
    df.info()
    print("\n--- NaN counts per column (after loading) ---")
    print(df.isnull().sum())

    processed_file = os.path.join(OUTPUT_DIR, "processed_data_with_resultant.csv")
    df.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}")

    sample_rate = 0
    # (Sample rate determination logic - same as before)
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
    if sample_rate == 0 and 'Tick' in df.columns:
        try:
            print("Tick column found, but sample rate inference from Tick is not fully implemented.")
        except Exception as e:
            print(f"Error processing 'Tick' for sample rate: {e}")
    if sample_rate == 0:
        try:
            manual_sample_rate = float(input("Enter sample rate in Hz (e.g., 2600.0): "))
            if manual_sample_rate <= 0: raise ValueError("Rate must be positive.")
            sample_rate = manual_sample_rate
        except ValueError as e:
            print(f"Invalid rate ({e}). Defaulting to 2600.0 Hz."); sample_rate = 2600.0
    print(f"Using sample rate: {sample_rate:.2f} Hz for FFT analysis.")

    # Plot time domain for X, Y, Z (converted), and Resultant
    plot_time_domain_all(df, OUTPUT_DIR, sample_rate_for_time_axis=sample_rate)

    print("\n--- Vibration Analysis for Balancing (using Resultant) ---")

    signal_name_for_analysis = 'Resultant'
    signal_data_for_analysis = df[signal_name_for_analysis].fillna(0).values

    if len(signal_data_for_analysis) == 0:
        print(f"{signal_name_for_analysis} has no data. Skipping analysis.");
        return

    # FFT Analysis for Resultant
    print(f"\n--- Processing FFT for {signal_name_for_analysis} ---")
    freqs, magnitude = compute_fft(signal_data_for_analysis, sample_rate)
    if len(freqs) > 0:
        peak_freqs, peak_mags = find_peak_frequencies(freqs, magnitude, num_peaks=10)
        print(f"\nTop {len(peak_freqs)} peak frequencies for {signal_name_for_analysis} (sorted by frequency):")
        if len(peak_freqs) > 0:
            for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags)): print(
                f"{i + 1}. {freq:.2f} Hz (mag: {mag:.6f})")
        else:
            print("No peaks found.")
        plot_fft_results(signal_name_for_analysis, freqs, magnitude, peak_freqs, peak_mags, OUTPUT_DIR)
        save_axis_fft_results_to_csv(signal_name_for_analysis, freqs, magnitude, peak_freqs, peak_mags, OUTPUT_DIR)
    else:
        print(f"FFT computation failed for {signal_name_for_analysis}.")

    # Polar Plot with Encoder for Resultant
    print(f"\nGenerating color-coded polar plot for {signal_name_for_analysis} using Encoder data...")
    angle_max_vib_deg = plot_vibration_polar(
        signal_name_for_analysis,
        df[signal_name_for_analysis],  # Pass the Series for the signal
        df['Encoder'],  # Pass the Series for the Encoder
        OUTPUT_DIR
    )

    if angle_max_vib_deg is not None:
        correction_angle = (angle_max_vib_deg + 180) % 360
        print(f"  Estimated 'heavy spot' based on {signal_name_for_analysis} is around: {angle_max_vib_deg:.1f}°")
        print(f"  Consider placing a trial weight for overall balance around: {correction_angle:.1f}°")
    else:
        print(f"  Could not determine angle of maximum {signal_name_for_analysis} vibration.")
    print("  NOTE: The *amount* of trial weight must be determined experimentally.")

    print(f"\nAll analysis results saved to '{OUTPUT_DIR}/' directory.")
    print("All plots have been displayed.")
    print("\nIMPORTANT FOR BALANCING:")
    print("1. The angles provided are ESTIMATES of the 'heavy spot' based on max observed resultant vibration.")
    print(
        "2. A correction trial weight should be placed approximately 180 DEGREES OPPOSITE the identified heavy spot angle.")
    print(
        "3. This script DOES NOT calculate the AMOUNT of weight. This is typically found by adding a small trial weight, observing the change in vibration, and then calculating the final correction, or through careful iteration.")
    print(
        "4. For complex systems, or if this doesn't resolve the issue, consult professional balancing resources or services.")


if __name__ == "__main__":
    main()