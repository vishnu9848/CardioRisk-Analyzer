import pandas as pd
import joblib, numpy as np
import shap

# Load the model and explainer
model = joblib.load('./Models/HAP_model.pkl')
explainer = joblib.load('./Models/HAP_explainer.pkl')

# Load data to get feature names
data = pd.read_csv('./Data/heart.csv')
features = data.drop('target', axis=1)
feature_names = features.columns

def generate_results(patient_data, model, explainer, feature_names):
    patient_data_df = pd.DataFrame([patient_data], columns=feature_names)

    risk_prob = model.predict_proba(patient_data_df)[0, 1]
    shap_values = explainer.shap_values(patient_data_df)
    feature_importance = dict(zip(feature_names, shap_values[0]))

    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]).sum() if isinstance(x[1], (list, np.ndarray)) else abs(x[1]), reverse=True)
    top_features = sorted_features[:3]

    suggestions = []
    important_factors = []
    feature_full_names = {    "cp"  :"Chest Pain Type", 
                               "trestbps" :"Resting Blood Pressure", 
                               "chol" :"Serum Cholesterol", 
                               "fbs" :"Fasting Blood Sugar > 120 mg/dl", 
                               "restecg" :"Resting ECG Results", 
                               "thalach" :"Maximum Heart Rate Achieved", 
                               "exang" :"Exercise Induced Angina", 
                               "oldpeak" :"ST Depression", 
                               "slope" :"Slope of Peak Exercise ST Segment", 
                               "ca" :"Number of Major Vessels", 
                               "thal" :"Thalassemia"
    }
    # Add suggestions based on influential features without displaying importance values
    for feature, importance in top_features:
        full_name = feature_full_names.get(feature, feature) # Get full name or default to feature name
        if isinstance(importance, (list, np.ndarray)):
            importance = importance.sum()
        if importance < 0:
            importance = abs(importance)  # Take absolute value for negative SHAP values
        important_factors.append(f"The feature '{full_name}' is significantly impacting your risk.")

    # Add detailed suggestions (unchanged)
    chol_value = patient_data[4]
    if chol_value > 240:
        suggestions.append('Your cholesterol level is high. Consider reducing it through a heart-healthy diet, regular exercise, and possibly medication.\n High cholesterol can lead to atherosclerosis and coronary artery disease.')
    elif chol_value > 200:
        suggestions.append('Your cholesterol level is borderline high. Maintaining a healthy diet and regular exercise can help manage this.\n You may be at risk of developing atherosclerosis.')
    elif chol_value < 130:
        suggestions.append('Your cholesterol level is low. It’s best to consult a doctor to ensure this isn’t indicative of an underlying condition.')

    age_value = patient_data[0]
    if age_value > 60:
        suggestions.append('At your age, regular health check-ups and a heart-healthy lifestyle are essential. You may be at an increased risk for heart disease, such as heart failure or coronary artery disease.')
    elif age_value > 40:
        suggestions.append('You are in a critical age range for heart disease. Regular monitoring of cholesterol, blood pressure, and lifestyle is important to prevent conditions like atherosclerosis or myocardial infarction.')

    cp_value = patient_data[2]
    if cp_value == 1:
        suggestions.append('You have experienced typical angina. It is recommended to get this evaluated by a cardiologist as it may indicate coronary artery disease.')
    elif cp_value == 2:
        suggestions.append('You have experienced atypical angina. Please monitor symptoms and consult your doctor, as it could be a sign of coronary microvascular disease.')
    elif cp_value == 3:
        suggestions.append('You have experienced non-anginal pain. While less concerning, monitoring is still important to rule out conditions like stable angina.')

    trestbps_value = patient_data[3]
    if trestbps_value > 140:
        suggestions.append('Your blood pressure is high. Regular monitoring and consultation with a doctor for lifestyle changes or medication are advised.\n High blood pressure can lead to hypertension and hypertensive heart disease.')
    elif trestbps_value < 90:
        suggestions.append('Your blood pressure is low. It’s recommended to consult with a doctor to rule out underlying issues, as it could lead to orthostatic hypotension or other conditions.')

    fbs_value = patient_data[5]
    if fbs_value == 1:
        suggestions.append('Your fasting blood sugar is high. This could be a sign of diabetes or prediabetes.\n Monitor blood sugar levels closely and consult your doctor, as it increases the risk of diabetic cardiomyopathy.')

    thalach_value = patient_data[7]
    if thalach_value < 120:
        suggestions.append('Your heart rate response to exercise is lower than normal.\n Discuss your cardiovascular fitness with a healthcare provider to rule out any issues like exercise-induced ischemia.')

    oldpeak_value = patient_data[9]
    if oldpeak_value > 2:
        suggestions.append('Significant ST depression detected. This may be a sign of ischemia. Further evaluation with a cardiologist is recommended.')

    ca_value = patient_data[11]
    if ca_value > 0:
        suggestions.append('A higher number of major vessels with calcification may indicate coronary artery disease.\n Please consult your doctor for a detailed examination.')

    thal_value = patient_data[12]
    if thal_value == 1:
        suggestions.append('You may have a fixed defect related to thalassemia. Consult a healthcare provider for further investigation.')
    elif thal_value == 2:
        suggestions.append('You may have a reversible defect. Follow-up with a healthcare provider is recommended for further evaluation.')

    return risk_prob * 100, suggestions, important_factors  # Multiply by 100 before returning


# Example usage
if __name__ == "__main__":

    high_risk_patient = [65, 1, 3, 150, 280, 1, 2, 120, 1, 2.3, 2, 2, 3]
    # Features: Age 65, Male, Chest Pain Type: Asymptomatic, Resting BP: 150, Cholesterol: 280, Fasting Blood Sugar > 120 mg/dl, 
    # Resting ECG: 2, Max Heart Rate: 120, Exercise Induced Angina: Yes, ST Depression: 2.3, Slope: 2, Major Vessels: 2, Thalassemia: 3

    moderate_risk_patient = [58, 1, 2, 130, 220, 0, 1, 145, 0, 1.4, 1, 0, 2]
    # Features: Age 58, Male, Chest Pain Type: Atypical Angina, Resting BP: 130, Cholesterol: 220, Fasting Blood Sugar <= 120 mg/dl,
    # Resting ECG: 1, Max Heart Rate: 145, Exercise Induced Angina: No, ST Depression: 1.4, Slope: 1, Major Vessels: 0, Thalassemia: 2

    low_risk_patient = [45, 0, 0, 110, 190, 0, 1, 160, 0, 0.5, 2, 0, 2]
    # Features: Age 45, Female, Chest Pain Type: Typical Angina, Resting BP: 110, Cholesterol: 190, Fasting Blood Sugar <= 120 mg/dl,
    # Resting ECG: 1, Max Heart Rate: 160, Exercise Induced Angina: No, ST Depression: 0.5, Slope: 2, Major Vessels: 0, Thalassemia: 2


    patient_data = high_risk_patient
    risk_prob, suggestions, important_fact = generate_results(patient_data, model, explainer, feature_names)
    print(f'Predicted Probability of Heart Attack: {risk_prob:.2f}%')
    print('Suggestions:')
    for suggestion in suggestions:
        print(f'- {suggestion}')
    for factor in important_fact:
        print(f'- {factor}')
