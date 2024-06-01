import pandas as pd

# Load the dataset
data = pd.read_csv(
    "/Users/albertasamoah/Desktop/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/data/census.csv"
)

# Display information about the dataset
print("Dataset Information:")
print(data.info())
print()

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display summary statistics for numerical features
print("Summary Statistics:")
print(data.describe())
print()
