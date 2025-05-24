import pandas as pd

# Define file path
file_path = r"C:\Users\Asus\Documents\Work\Transfermarket\archive\final_cleaned_players_data.csv"

# Load dataset
df = pd.read_csv(file_path)

# Print all column names vertically
print("Column names in the dataset:")
for col in df.columns:
    print(col)
