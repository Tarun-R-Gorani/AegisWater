import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="AegisWater AI", page_icon="🛡️")

# 2. Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('models/water_model.joblib')

model = load_model()

# 3. Sidebar Header
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3144/3144467.png", width=100)
st.sidebar.title("AegisWater Settings")
st.sidebar.info("This system uses a Random Forest model to predict water safety based on 9 chemical parameters.")

# 4. Main UI
st.title("🛡️ AegisWater: Smart Potability Classifier")
st.markdown("---")

st.subheader("Enter Water Chemical Composition")
col1, col2, col3 = st.columns(3)

# Organizing the 9 inputs into 3 columns
with col1:
    ph = st.number_input("pH level (0-14)", 0.0, 14.0, 7.0)
    hardness = st.number_input("Hardness (mg/L)", 40.0, 330.0, 196.0)
    solids = st.number_input("Solids (ppm)", 300.0, 60000.0, 20000.0)

with col2:
    chloramines = st.number_input("Chloramines (ppm)", 0.0, 15.0, 7.1)
    sulfate = st.number_input("Sulfate (mg/L)", 100.0, 500.0, 333.0)
    conductivity = st.number_input("Conductivity (μS/cm)", 100.0, 800.0, 426.0)

with col3:
    organic_carbon = st.number_input("Organic Carbon (ppm)", 0.0, 30.0, 14.0)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", 0.0, 130.0, 66.0)
    turbidity = st.number_input("Turbidity (NTU)", 1.0, 7.0, 3.9)

st.markdown("---")

# 5. Prediction Logic
if st.button("Analyze Water Safety", use_container_width=True):
    # Prepare the feature array
    features = np.array([[ph, hardness, solids, chloramines, sulfate, 
                          conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    # Get prediction and probability
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    # Display Results
    if prediction == 1:
        st.success(f"### ✅ VERDICT: POTABLE (SAFE)")
        st.write(f"The model is **{probability*100:.1f}%** confident this water is safe to drink.")
        st.balloons()
    else:
        st.error(f"### 🚨 VERDICT: NOT POTABLE (UNSAFE)")
        st.write(f"The model is **{(1-probability)*100:.1f}%** confident this water is contaminated.")

    # Show a progress bar for safety confidence
    st.write("Safety Confidence Level:")
    st.progress(probability)