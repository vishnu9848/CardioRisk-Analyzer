import streamlit as st
import pandas as pd
import joblib, time, textwrap
from implement import generate_results






st.set_page_config(layout="wide")


# Create the form
def form():
    # Set the title
    st.title("CardioRisk Analyzer\n Personalized Heart Health Predictions")

    with st.form("patient_details_form"):
        name = st.text_input("Name", help="Please enter your name") # Proceed only if the name is provided 
        
        age = st.number_input("Age (in years)", value=45, step=1)
        sex = st.radio("Sex", ["Male", "Female"], index=1)
        cp = st.selectbox("Chest Pain Type", [
            "Typical Angina (1): Describes chest pain that is related to exertion and is predictable.",
            "Atypical Angina (2): Describes chest pain that is not related to exertion or that occurs with varied stimuli.",
            "Non-anginal Pain (3): Describes chest pain that is not related to the heart.",
            "Asymptomatic (4): Chest pain that is silent and doesn't cause symptoms."
        ], index=2)
        trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", value=110, step=1)
        chol = st.number_input("Serum Cholesterol (in mg/dl)", value=190, step=1)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", ["True", "False"], index=1)
        restecg = st.number_input("Resting ECG Results (0-2)", value=1, step=1)
        thalach = st.number_input("Maximum Heart Rate Achieved", value=160, step=1)
        exang = st.radio("Exercise Induced Angina (1 = yes; 0 = no)", ["Yes", "No"], index=1)
        oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", value=0.5, format="%.1f")
        slope = st.number_input("Slope of the Peak Exercise ST Segment (0-2)", value=2, step=1)
        ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy (0-3)", value=0, step=1)
        thal = st.number_input("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)", value=2, step=1)

        # Submit button
        submitted = st.form_submit_button("Submit")

        patient_info = {"Parameter": ["Age", "Sex", "Chest Pain Type", "Resting Blood Pressure", "Serum Cholesterol", "Fasting Blood Sugar > 120 mg/dl", "Resting ECG Results", "Maximum Heart Rate Achieved", "Exercise Induced Angina", "ST Depression", "Slope of Peak Exercise ST Segment", "Number of Major Vessels", "Thalassemia"],
                            "Value": [age, sex, cp.split(':')[0], trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
                            }   

        # Convert dictionary to DataFrame
        df = pd.DataFrame(patient_info)

        # Handle form submission
        if submitted:
            # Convert selections to appropriate values
            sex_value = 1 if sex == "Male" else 0
            cp_value = 1 if cp.startswith("Typical") else 2 if cp.startswith("Atypical") else 3 if cp.startswith("Non-anginal") else 4
            fbs_value = 1 if fbs == "True" else 0
            exang_value = 1 if exang == "Yes" else 0
            
            patient_data = [
                int(age), sex_value, cp_value, int(trestbps), int(chol), 
                fbs_value, int(restecg), int(thalach), 
                exang_value, float(oldpeak), int(slope), int(ca), int(thal)
            ]
            
            st.session_state['name'] = name
            st.session_state['patient_data'] = patient_data
            st.session_state['df'] = df
            if not name: 
                st.warning("Please enter your name to proceed.")
            else:
                st.session_state['page'] = 'Result'



def result():
    if 'name' in st.session_state:
        st.title(f"{st.session_state['name']} Details")
    if 'df' in st.session_state and 'patient_data' in st.session_state:
        df = st.session_state['df']
        patient_data = st.session_state['patient_data']

        # Create a single container to wrap everything
        container = st.container()

        with container:
            st.subheader("Patient Data")
            # Adjust the table size to fit within A4 width
            st.dataframe(df, width=700, height=493, hide_index=True)

            model = joblib.load('./Models/HAP_model.pkl')
            explainer = joblib.load('./Models/HAP_explainer.pkl')
            data = pd.read_csv('./Data/heart.csv')
            features = data.drop('target', axis=1)
            feature_names = features.columns

            try:
                start = time.time()
                risk_prob, suggestions, important_factors = generate_results(patient_data=patient_data, model=model, explainer=explainer, feature_names=feature_names)
                end = time.time()
                result_time = round(end - start, 2)
                st.write(f"Execution Time: {result_time} sec")

                st.subheader("Risk Probability")

                # Using Streamlit's progress bar with conditional coloring and adjusted width
                color = 'red' if risk_prob > 75 else 'orange' if risk_prob > 50 else 'green'
                st.markdown(f"""
                    <div style="position:relative; width: 50%; height: 40px; background-color: lightgray; border: 1px solid #ddd; float: left; margin-left: 0;">
                        <div style="position:absolute; width: {risk_prob}%; height: 100%; background-color: {color};"></div>
                        <div style="position:absolute; width: 100%; height: 100%; text-align: center; line-height: 40px; color: black;">{int(risk_prob)}% - {'Critical' if risk_prob > 75 else 'Elevated' if risk_prob > 50 else 'Normal'}</div>
                    </div>
                """, unsafe_allow_html=True)
                

                
                st.subheader("Crucial Health Indicators")
                for imp_fact in important_factors:
                    st.markdown(f"* {imp_fact}")  # Added a space after the asterisk
            
                st.subheader("Suggestions")
                for suggestion in suggestions: 
                    # Split the suggestion into lines of up to 80 characters
                    wrapped_suggestion = "\n".join(textwrap.wrap(suggestion, width=80)) 
                    st.markdown(f"* {wrapped_suggestion}")
                
            except NameError as e:
                st.error(f"Error: {e}")


# Main Loop
if 'page' not in st.session_state:
    st.session_state['page'] = 'Form'

if st.session_state['page'] == 'Form':
    form()
elif st.session_state['page'] == 'Result':
    result()
    
