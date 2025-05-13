import numpy as np
import csv

"""
THIS ONE TAKES A DATASET AND GENERATES MOCK ENCODER DATA
ACCORDINGLY
"""

# --- Configuration ---
# !!!! USER: Please change 'vibration_data.csv' to the actual name of your CSV file !!!!
input_filename = input("Enter csv file name: ")
output_filename = 'updated_vibration_data.csv' # Optional: for saving the output
# --- End Configuration ---

header = []
data_list = []

# Step 1: Read the CSV data from the specified file
try:
    with open(input_filename, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read the header row
        data_list = list(reader)  # Read the rest of the data
except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found.")
    print("Please make sure the file exists in the same directory as the script, or provide the full path.")
    exit()
except StopIteration:
    print(f"Error: The CSV file '{input_filename}' is empty or does not contain a header row.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the file '{input_filename}': {e}")
    exit()

if not data_list:
    print(f"Error: No data rows found in '{input_filename}' after the header.")
    exit()

# Step 2: Determine column data types and prepare for NumPy structured array
try:
    # Ensure 'Encoder' column exists, which is crucial for the script's logic
    if 'Encoder' not in header:
        raise ValueError("'Encoder' column not found in CSV header.")
except ValueError as e:
    print(f"Error: {e}")
    exit()

dt_types = []
for col_name in header:
    if col_name == 'Timestamp':
        # Adjust 'U23' if your timestamps are longer/shorter
        dt_types.append((col_name, 'U23'))
    else:
        # Assume other relevant columns are numeric (float)
        dt_types.append((col_name, float))

# Step 3: Convert data to the correct types and create the NumPy array
typed_data_list = []
for row_idx, row in enumerate(data_list):
    if len(row) != len(header):
        print(f"Warning: Row {row_idx + 2} (1-based, including header) has {len(row)} values, but header has {len(header)} columns. Skipping this row.")
        continue
    new_row_tuple = []
    for col_idx, item in enumerate(row):
        col_name = header[col_idx]
        if col_name == 'Timestamp':
            new_row_tuple.append(item)
        else:
            try:
                new_row_tuple.append(float(item))
            except ValueError:
                print(f"Warning: Could not convert '{item}' to float for column '{col_name}' in row {row_idx + 2}. Using np.nan.")
                new_row_tuple.append(np.nan)
    typed_data_list.append(tuple(new_row_tuple))

if not typed_data_list:
    print("Error: No valid data rows could be processed from the CSV after type conversion.")
    exit()

numpy_data = np.array(typed_data_list, dtype=dt_types)

# Step 4: Fill the 'Encoder' column according to the logic
if len(numpy_data) > 0:
    current_encoder_value = 0.0
    for i in range(len(numpy_data)):
        if i == 0:
            # Set the first encoder value to 0 as per the logic
            numpy_data[i]['Encoder'] = 0.0
            current_encoder_value = 0.0
        else:
            # Increment by 0.68 and take modulus 360
            current_encoder_value = (current_encoder_value + 0.68) % 360
            numpy_data[i]['Encoder'] = round(current_encoder_value, 5) # Rounded for precision

# Step 5: Print the updated NumPy array (or save it)
print(f"\nUpdated Data (from '{input_filename}'):")
print(",".join(header)) # Print header

for row in numpy_data:
    formatted_row_values = []
    for col_name in header:
        value = row[col_name]
        if isinstance(value, float):
            # Format float columns for consistent output
            if col_name == 'Tick': # Assuming Tick should be integer-like if it's a whole number
                 formatted_row_values.append(str(int(value)) if value.is_integer() else f"{value:.2f}") # Example: Tick as int or 2 decimal places
            elif col_name in ['X', 'Y', 'Z', 'Encoder']:
                 formatted_row_values.append(f"{value:.5f}") # Higher precision for these
            else:
                 formatted_row_values.append(str(value))
        else: # For Timestamp (string) or other non-float types
            formatted_row_values.append(str(value))
    print(",".join(formatted_row_values))

# Optional: Save the updated data to a new CSV file
# Remove the triple quotes to enable saving

try:
    with open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header
        for row_data in numpy_data:
            # Write out in the same order as the original header
            writer.writerow([row_data[field] for field in header])
    print(f"\nUpdated data successfully saved to '{output_filename}'")
except Exception as e:
    print(f"\nAn error occurred while saving the file '{output_filename}': {e}")