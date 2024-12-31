import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title='Autism Prediction App', layout='centered')

st.title('Autism Prediction Tool')
st.write("This tool predicts the likelihood of autism based on input features.")

rfc = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

def predict(A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score,
       A7_Score, A8_Score, A9_Score, A10_Score, age, gender,
       ethnicity, jaundice, austim, contry_of_res, used_app_before,
       result, relation):
    
    gender = encoder['gender'].transform([gender])[0]
    ethnicity = encoder['ethnicity'].transform([ethnicity])[0]
    jaundice = encoder['jaundice'].transform([jaundice])[0]
    austim = encoder['austim'].transform([austim])[0]
    contry_of_res = encoder['contry_of_res'].transform([contry_of_res])[0]
    used_app_before = encoder['used_app_before'].transform([used_app_before])[0]
    relation = encoder['relation'].transform([relation])[0]
    
    X = np.array([A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score,
                  A7_Score, A8_Score, A9_Score, A10_Score, age, gender,
                  ethnicity, jaundice, austim, contry_of_res, used_app_before,
                  result, relation])
    X = X.reshape(1, -1)
    
    return rfc.predict(X)[0]

st.header("Survey Questions")
scores = []
for i in range(1, 11):
    scores.append(st.selectbox(f"A{i}_Score", [0, 1], index=0))

st.header("Personal Information")
age = st.number_input('Age', min_value=1, max_value=100, value=25)
gender = st.radio('Gender', ['m', 'f'], horizontal=True)
ethnicity = st.selectbox('Ethnicity', ['White-European', 'Middle Eastern', 'Pasifika', 'Black', 'Others', 'Hispanic', 'Asian', 'Turkish', 'South Asian', 'Latino'])
jaundice = st.radio('Jaundice', ['yes', 'no'], horizontal=True)
austim = st.radio('Autism in Family', ['yes', 'no'], horizontal=True)
contry_of_res = st.text_input('Country of Residence')

st.header("App Usage & Results")
used_app_before = st.radio('Used App Before', ['yes', 'no'], horizontal=True)
result = st.number_input('Result')
relation = st.selectbox('Relation', ['Self', 'Others'])

if st.button('Predict Autism Risk'):
    prediction = predict(*scores, age, gender, ethnicity, jaundice, austim, contry_of_res, used_app_before, result, relation)
    if prediction == 1:
        st.error("High risk of autism detected. Please consult a professional for further diagnosis.")
    else:
        st.success("Low risk of autism detected. No immediate concerns.")
