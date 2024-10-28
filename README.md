# Heart Disease Prediction and Analysis

## Project Overview

This project involves building a machine learning model to predict the likelihood of heart disease based on patient data. The model uses a Random Forest Classifier to predict whether a patient is likely to have heart disease. Additionally, the project includes generating suggestions based on the prediction and interpreting the model’s results using SHAP (SHapley Additive exPlanations) values.

## Data

The project uses three datasets:
1. **Heart Disease Prediction Dataset** (`Heart_Disease_Prediction.csv`): Contains patient data with features and target labels ("Presence" or "Absence") for heart disease.
2. **Heart Disease Features Dataset** (`heart_disease_features.csv`): Contains patient data with various features but no target labels.
3. **Heart Disease Target Dataset** (`heart_disease_target.csv`): Contains target labels corresponding to the features in the second dataset.

### Data Preprocessing

1. **Column Renaming**: Columns in datasets are renamed to match for consistency.
2. **Encoding Labels**: The target labels "Presence" and "Absence" are encoded as 1 and 0, respectively.
3. **Combining Data**: The datasets are combined to create a unified training file for model training.

## Model Training

The Random Forest Classifier is used for training the model. The following steps were followed:
1. **Load Data**: Load and preprocess the data from the CSV files.
2. **Split Data**: Divide the data into training and testing sets.
3. **Train Model**: Train the Random Forest model on the training data.
4. **Save Model**: Save the trained model and SHAP explainer for future use.

## Model Evaluation

The model’s performance is evaluated using accuracy, precision, recall, F1 score, and ROC AUC score.

## Prediction and Suggestions

The trained model can be used to predict the probability of heart disease for new patient data. Suggestions are generated based on the prediction and feature values. SHAP values are used to understand the impact of different features on the prediction.

## Usage

1. **Train the Model**: Run `train.py` to train and save the model.
2. **Generate Predictions**: Run `implement.py` with new patient data to get predictions and suggestions.

## Files

- `train.py`: Script to train the model and save it.
- `implement.py`: Script to generate predictions and suggestions using the trained model.
- `Heart_Disease_Prediction.csv`: Dataset with features and target labels.
- `heart_disease_features.csv`: Dataset with patient features.
- `heart_disease_target.csv`: Dataset with target labels.
- `heart_disease_model.pkl`: Saved trained model.
- `heart_disease_explainer.pkl`: Saved SHAP explainer.

## Requirements

To install the necessary packages, use the `requirements.txt` file provided.

## License

This project is licensed under the MIT License.
