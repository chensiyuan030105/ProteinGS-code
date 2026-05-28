import pandas as pd

# Load the CSV file
csv_file_path = './dataset/supported_data/af3_metadata_antibody_antigen.csv'  # Replace with your actual file path
df = pd.read_csv(csv_file_path)

# Extract the first column and remove duplicates
unique_first_column_values = df.iloc[:, 0].unique()

# Print the result as a single column without quotes
print("Unique values in the first column:")
print("\n".join(map(str, unique_first_column_values)))

# Extract the first three columns
first_three_columns = df.iloc[:, :3]

# Print the first three columns with comma separation
print("First three columns:")
print(first_three_columns.to_csv(index=False, header=False, sep=',', line_terminator='\n'))

