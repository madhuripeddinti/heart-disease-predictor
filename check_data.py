
import pandas as pd

# Load the heart disease dataset
data = pd.read_csv("heart.csv")

# Display first 5 rows
print("First 5 rows:")
print(data.head())

# Display dataset information
print("\nDataset information:")
print(data.info())

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Display number of rows and columns
print("\nDataset shape:")
print(data.shape)

