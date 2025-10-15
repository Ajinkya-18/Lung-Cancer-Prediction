import sys
import os
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_model


st.set_page_config(
    page_title="Lung Cancer Predictor", 
    page_icon="🩺", 
    layout='wide'
)

st.warning(
    '''
    **Disclaimer:** This app is an educational tool and not a substitute for professional medical advice, diagnosis, or treatment. 
    Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    '''
)


try:
    model = load_model('models/random_forest_classifier_fitted.joblib')
    col_transformer = load_model('models/col_transformer_fitted.joblib')
    target_encoder = load_model('models/ordinal_encoder_fitted.joblib')

except Exception as e:
    st.error(f"Error loadingthe models: {e}")
    st.stop()


st.title("Lung Cancer Susceptibility Prediction App")
st.markdown("""This app predicts the likelihood of lung cancer based on various lifestyle and environmental factors. 
            Please input the patient's information on the left sidebar.""")

st.sidebar.header("Patient Data")

def user_input_features():
    age = st.sidebar.slider('Age', 20, 90, 45)
    air_pollution = st.sidebar.slider('Air Pollution Exposure (1-8)', 1, 8, 3)
    alcohol_use = st.sidebar.slider('Alcohol Use (1-8)', 1, 8, 3)
    dust_allergy = st.sidebar.slider('Dust Allergy (1-8)', 1, 8, 3)
    occupational_hazards = st.sidebar.slider('Occupational Hazards (1-8)', 1, 8, 3)
    genetic_risk = st.sidebar.slider('Genetic Risk (1-7)', 1, 7, 3)
    chronic_lung_disease = st.sidebar.slider('Chronic Lung Disease (1-7)', 1, 7, 3)
    balanced_diet = st.sidebar.slider('Balanced Diet (1-7)', 1, 7, 3)
    obesity = st.sidebar.slider('Obesity (1-7)', 1, 7, 3)
    smoking = st.sidebar.slider('Smoking (1-8)', 1, 8, 3)
    passive_smoker = st.sidebar.slider('Passive Smoker (1-8)', 1, 8, 3)
    chest_pain = st.sidebar.slider('Chest Pain (1-9)', 1, 9, 3)
    coughing_of_blood = st.sidebar.slider('Coughing of Blood (1-9)', 1, 9, 4)
    fatigue = st.sidebar.slider('Fatigue (1-9)', 1, 9, 3)
    weight_loss = st.sidebar.slider('weight_loss (1-8)', 1, 8, 3)
    shortness_of_breath = st.sidebar.slider('Shortness of Breath (1-9)', 1, 9, 3)
    wheezing = st.sidebar.slider('Wheezing (1-8)', 1, 8, 3)
    swallowing_difficulty = st.sidebar.slider('Swallowing Difficulty (1-8)', 1, 8, 3)
    clubbing_of_finger_nails = st.sidebar.slider('Clubbing of Finger Nails (1-9)', 1, 9, 3)
    frequent_cold = st.sidebar.slider('Frequent Cold (1-7)', 1, 7, 3)
    dry_cough = st.sidebar.slider('Dry Cough (1-7)', 1, 7, 3)
    snoring = st.sidebar.slider('Snoring (1-7)', 1, 7, 3)

    data = {'Age': [age], 
            'Air Pollution': [air_pollution],
            'Alcohol use': [alcohol_use],
            'Dust Allergy': [dust_allergy],
            'OccuPational Hazards': [occupational_hazards],
            'Genetic Risk': [genetic_risk],
            'chronic Lung Disease': [chronic_lung_disease],
            'Balanced Diet': [balanced_diet],
            'Obesity': [obesity],
            'Smoking': [smoking],
            'Passive Smoker': [passive_smoker],
            'Chest Pain': [chest_pain],
            'Coughing of Blood': [coughing_of_blood],
            'Fatigue': [fatigue],
            'Weight Loss': [weight_loss],
            'Shortness of Breath': [shortness_of_breath],
            'Wheezing': [wheezing],
            'Swallowing Difficulty': [swallowing_difficulty],
            'Clubbing of Finger Nails': [clubbing_of_finger_nails],
            'Frequent Cold': [frequent_cold],
            'Dry Cough': [dry_cough],
            'Snoring': [snoring]
            }

    features = pd.DataFrame(data, index=[0])

    return features

input_df = user_input_features()

st.subheader("Patient Input Parameters")
st.write(input_df)

if st.button('Predict Susceptibility'):
    try: 
        processed_input = col_transformer.transform(input_df)
        prediction_encoded = model.predict(processed_input)

        prediction = target_encoder.inverse_transform(prediction_encoded.reshape(-1, 1))[0][0]

        st.success(f"Prediction Complete!")

        if prediction.lower() == 'high':
            st.error(f"Predicted Susceptibility: **{prediction}**")
        elif prediction.lower() == 'medium':
            st.warning(f"Predicted Susceptibility: **{prediction}**")
        else:
            st.success(f"Predicted Susceptibility: **{prediction}**")

    except Exception as e:
        st.error(f"An error occurred: {e}")




