import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch the dataset
heart_disease = fetch_ucirepo(id=45)

# Extract the features and target
features = heart_disease.data.features
target = heart_disease.data.targets

# Save features and target to CSV files
features_csv_path = './Data/heart_disease_features.csv'
target_csv_path = './Data/heart_disease_target.csv'

# Save features to CSV
features.to_csv(features_csv_path, index=False)
print(f'Features data saved to {features_csv_path}')

# Save target to CSV
target.to_csv(target_csv_path, index=False)
print(f'Target data saved to {target_csv_path}')
