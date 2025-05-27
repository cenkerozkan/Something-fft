import csv

def clean_encoder_csv(input_file, output_file):
    """
    Reads a CSV file and writes a new CSV file that starts with the first row
    where the 'Encoder' column is 0, keeping all subsequent rows.

    Args:
        input_file: Path to the input CSV file.
        output_file: Path to the cleaned output CSV file.
    """
    with open(input_file, 'r', newline='') as fin, open(output_file, 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        writer.writerow(header)

        encoder_idx = header.index('Encoder')

        # Skip rows until first Encoder == 0 is found
        found_zero = False
        for row in reader:
            if not found_zero:
                try:
                    encoder_value = float(row[encoder_idx])
                except ValueError:
                    continue  # Skip rows with invalid data
                if encoder_value == 0:
                    found_zero = True
                    writer.writerow(row)
            else:
                writer.writerow(row)

if __name__ == "__main__":
    # Example usage
    clean_encoder_csv(input("Enter file name: "), 'output.csv')