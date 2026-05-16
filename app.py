import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. Page Configuration & Aquatic Theme
st.set_page_config(page_title="AegisWater Explorer", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

# Inject Custom CSS for the "Blue Water" Vibe
st.markdown("""
    <style>
    .stApp {
        background-color: #0B132B;
        color: #E0FFFF;
    }
    h1, h2, h3 {
        color: #48CAE4 !important;
    }
    .stButton>button {
        background-color: #0077B6;
        color: white;
        border: 1px solid #48CAE4;
        border-radius: 5px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #48CAE4;
        color: #0B132B;
        border: 1px solid #0077B6;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Database & Dictionaries
WHO_LIMITS = {
    'pH Level': {'key': 'ph', 'max': 8.5, 'unit': 'pH units'},
    'Hardness': {'key': 'Hardness', 'max': 200.0, 'unit': 'mg/L'},
    'Solids (TDS)': {'key': 'Solids', 'max': 1000.0, 'unit': 'ppm'},
    'Chloramines': {'key': 'Chloramines', 'max': 4.0, 'unit': 'ppm'},
    'Sulfate': {'key': 'Sulfate', 'max': 250.0, 'unit': 'mg/L'},
    'Conductivity': {'key': 'Conductivity', 'max': 400.0, 'unit': 'μS/cm'},
    'Organic Carbon': {'key': 'Organic_Carbon', 'max': 4.0, 'unit': 'ppm'},
    'Trihalomethanes': {'key': 'Trihalomethanes', 'max': 80.0, 'unit': 'μg/L'},
    'Turbidity': {'key': 'Turbidity', 'max': 5.0, 'unit': 'NTU'}
}

CONTAMINATION_SOURCES = {
    'ph': "Industrial chemical dumping or agricultural alkaline runoff.",
    'Hardness': "Geological weathering of deep underground limestone deposits.",
    'Solids': "Sewage discharge, industrial wastewater, or heavy road-salting.",
    'Chloramines': "Over-chlorination by water treatment centers.",
    'Sulfate': "Runoff from mining industries or paper mills.",
    'Conductivity': "Industrial discharges or saltwater intrusion.",
    'Organic_Carbon': "Decaying organic vegetation or manure runoff.",
    'Trihalomethanes': "Chlorine reacting with natural organic matter.",
    'Turbidity': "Soil erosion, stormwater runoff, or high algae blooms."
}

# 3. Load Model Engine
@st.cache_resource
def load_model_bundle():
    return joblib.load('rf_model.joblib')

bundle = load_model_bundle()

# --- Custom UI Component for Dual Inputs ---
def dual_input(label, min_val, max_val, default_val, step, key_prefix):
    """Creates a synchronized slider and number input side-by-side."""
    # Initialize session state for this parameter if it doesn't exist
    if f"{key_prefix}_val" not in st.session_state:
        st.session_state[f"{key_prefix}_val"] = float(default_val)

    # Callbacks to sync the slider and the number box
    def sync_from_slider():
        st.session_state[f"{key_prefix}_val"] = st.session_state[f"{key_prefix}_slider"]
    
    def sync_from_num():
        st.session_state[f"{key_prefix}_val"] = st.session_state[f"{key_prefix}_num"]

    st.write(label)
    col1, col2 = st.columns([3, 1.5])
    with col1:
        st.slider(
            label, min_value=float(min_val), max_value=float(max_val), 
            value=float(st.session_state[f"{key_prefix}_val"]), 
            step=float(step), key=f"{key_prefix}_slider", 
            on_change=sync_from_slider, 
            label_visibility="collapsed"
        )
    with col2:
        st.number_input(
            label, min_value=float(min_val), max_value=float(max_val), 
            value=float(st.session_state[f"{key_prefix}_val"]), 
            step=float(step), key=f"{key_prefix}_num", 
            on_change=sync_from_num, 
            label_visibility="collapsed"
        )
    return st.session_state[f"{key_prefix}_val"]

# 4. Main UI Layout
st.title("🌊 AegisWater AI: Quality Explorer")
st.markdown("Interactive parameter radar and predictive safety diagnostics.")

# Create a sleek sidebar for inputs
with st.sidebar:
    st.header("🎛️ Sample Telemetry")
    st.caption("Use the slider for estimates, or type exact values.")
    st.markdown("---")
    
    # Implementing the dual inputs
    ph = dual_input("pH Level", 0.0, 14.0, 7.0, 0.1, "ph")
    hardness = dual_input("Hardness (mg/L)", 40.0, 350.0, 195.0, 1.0, "hard")
    solids = dual_input("Solids / TDS (ppm)", 100.0, 50000.0, 20827.0, 10.0, "solids")
    chloramines = dual_input("Chloramines (ppm)", 0.0, 15.0, 7.1, 0.1, "chlor")
    sulfate = dual_input("Sulfate (mg/L)", 50.0, 500.0, 353.0, 1.0, "sulf")
    conductivity = dual_input("Conductivity (μS/cm)", 100.0, 800.0, 426.0, 1.0, "cond")
    organic_carbon = dual_input("Organic Carbon (ppm)", 0.0, 30.0, 14.0, 0.1, "org")
    trihalomethanes = dual_input("Trihalomethanes (μg/L)", 0.0, 130.0, 68.0, 1.0, "trihalo")
    turbidity = dual_input("Turbidity (NTU)", 0.0, 10.0, 9.3, 0.1, "turb")
    
    st.markdown("---")
    analyze_btn = st.button("Deploy Analysis 🚀", use_container_width=True)

# 5. Core Application Logic
if analyze_btn:
    # --- Prediction Engine ---
    raw_features = [ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]
    ph_dev = abs(ph - 7.5)
    sulfate_dev = abs(sulfate - 250.0)
    
    # Process through pipeline
    final_input_row = np.array([raw_features + [ph_dev, sulfate_dev]])
    scaler_art = bundle['scaler']
    classifier_art = bundle['classifier']
    
    scaled_input = scaler_art.transform(final_input_row)
    prediction = classifier_art.predict(scaled_input)[0]
    probability = classifier_art.predict_proba(scaled_input)[0][1]

    # --- Data Normalization for Graphs ---
    parameters = list(WHO_LIMITS.keys())
    raw_values = raw_features
    normalized_values = [(val / WHO_LIMITS[param]['max']) * 100 for val, param in zip(raw_values, parameters)]
    
    col_graphs, col_diagnostics = st.columns([2, 1])
    
    with col_graphs:
        # --- RADAR CHART ---
        st.subheader("PARAMETER RADAR")
        fig_radar = go.Figure()
        
        # WHO Target Polygon (100% boundary)
        fig_radar.add_trace(go.Scatterpolar(
            r=[100]*9 + [100],
            theta=parameters + [parameters[0]],
            fill='toself',
            name='WHO Limit',
            line_color='#00B4D8',
            fillcolor='rgba(0, 180, 216, 0.1)'
        ))
        
        # Actual Sample Polygon
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized_values + [normalized_values[0]],
            theta=parameters + [parameters[0]],
            fill='toself',
            name='Sample',
            line_color='#FF4D4D',
            fillcolor='rgba(255, 77, 77, 0.3)'
        ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(max(normalized_values), 150)])),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0FFFF'),
            height=400,
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # --- BAR CHART (Test Tube style) ---
        st.subheader("PARAMETER VS WHO LIMIT")
        colors = ['#FF4D4D' if val > 100 else '#00B4D8' for val in normalized_values]
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=parameters, 
                y=normalized_values, 
                marker_color=colors,
                text=[f"{val:.1f} {WHO_LIMITS[p]['unit']}" for val, p in zip(raw_values, parameters)],
                textposition='outside'
            )
        ])
        
        # Add the 100% threshold line
        fig_bar.add_hline(y=100, line_dash="dash", line_color="#FFB703", annotation_text="WHO Limit (100%)", annotation_font_color="#FFB703")
        
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0FFFF'),
            yaxis=dict(title="% of WHO Limit", gridcolor='rgba(255,255,255,0.1)'),
            height=300,
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- DIAGNOSTICS PANEL ---
    with col_diagnostics:
        st.subheader("SYSTEM VERDICT")
        if prediction == 1:
            st.success(f"### ✅ POTABLE\nSafety Confidence: **{probability*100:.1f}%**")
            st.balloons()
        else:
            st.error(f"### 🚨 CONTAMINATED\nRisk Assessment: **{(1-probability)*100:.1f}%**")
        
        st.markdown("---")
        st.subheader("Critical Anomalies")
        
        anomalies_found = False
        for param, val, norm in zip(parameters, raw_values, normalized_values):
            if norm > 100:
                anomalies_found = True
                key = WHO_LIMITS[param]['key']
                st.warning(f"**{param}**\n\nReading: {val} {WHO_LIMITS[param]['unit']}\n\n*Source: {CONTAMINATION_SOURCES[key]}*")
        
        if not anomalies_found:
            st.info("All parameters are within WHO physiological safety baselines.")
else:
    st.info("👈 Adjust parameters in the left sidebar and deploy the analysis to view the environmental dashboard.")
