import pandas as pd

# Load DataFrames
prediction_df = pd.read_csv('./Data/Heart_Disease_Prediction.csv')
features_df = pd.read_csv('./Data/heart_disease_features.csv')

# Rename columns in prediction_df to match features_df
prediction_df.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Encode target variable
prediction_df['target'] = prediction_df['target'].map({'Presence': 1, 'Absence': 0})

# Verify the updated DataFrame
print("Updated Prediction DataFrame Columns:", prediction_df.columns)
print("Features DataFrame Columns:", features_df.columns)

# Save the modified DataFrame
prediction_df.to_csv('./Data/Heart_Disease_Prediction_Encoded.csv', index=False)
