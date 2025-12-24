# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# ===========================
# Page Configuration
# ===========================
st.set_page_config(
    page_title="COPD Mortality Prediction Model",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit footer and menu for a professional look
hide_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_style, unsafe_allow_html=True)

# ===========================
# Resource Loading
# ===========================
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('rf_copd_model.pkl')
        background_data = joblib.load('background_data_for_shap.pkl')
        # Initialize Explainer
        explainer = shap.TreeExplainer(model)
        # Calculate SHAP values for Class 1 (Mortality)
        shap_values_full = explainer.shap_values(background_data)

        # Dimension compatibility handling
        if isinstance(shap_values_full, list):
            shap_v = shap_values_full[1]
        elif len(shap_values_full.shape) == 3:
            shap_v = shap_values_full[:, :, 1]
        else:
            shap_v = shap_values_full

        return model, background_data, shap_v
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        st.stop()

model, background_data, shap_v = load_resources()
feature_names = background_data.columns.tolist()

# ===========================
# Sidebar: Input Panel
# ===========================
st.sidebar.header("📋 Patient Clinical Parameters")

def user_input_features():
    st.sidebar.subheader("Categorical Variables (0: No, 1: Yes)")
    col_cat1, col_cat2 = st.sidebar.columns(2)
    with col_cat1:
        crrt = st.selectbox("CRRT", options=[0, 1])
        aki = st.selectbox("AKI", options=[0, 1])
    with col_cat2:
        pneumonia = st.selectbox("Pneumonia", options=[0, 1])

    st.sidebar.subheader("Physiological & Biochemical Indicators")
    age = st.sidebar.slider("Age (years)", 18, 100, 65)

    c1, c2 = st.sidebar.columns(2)
    with c1:
        hr = st.slider("Heart Rate (bpm)", 30, 200, 85)
        rr = st.slider("Respiratory Rate (bpm)", 5, 60, 20)
        sapsii = st.number_input("SAPS II Score", 0, 150, 40)
        ag = st.number_input("Anion Gap (mEq/L)", 0.0, 50.0, 12.0)
    with c2:
        ph = st.slider("Arterial pH", 6.8, 7.8, 7.35)
        lactate = st.number_input("Lactate (mmol/L)", 0.0, 25.0, 2.0)
        bun = st.number_input("BUN (mg/dL)", 0.0, 150.0, 25.0)
        cl = st.number_input("Chloride (mEq/L)", 50.0, 150.0, 105.0)

    inr = st.sidebar.number_input("INR", 0.5, 10.0, 1.2)

    data = {
        'CRRT': crrt, 'Age': age, 'AKI': aki, 'AG': ag, 'Cl': cl,
        'INR': inr, 'Lactate': lactate, 'pH': ph, 'BUN': bun,
        'HR': hr, 'RR': rr, 'Pneumonia': pneumonia, 'SAPSII': sapsii
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
input_df = input_df[feature_names]

# ===========================
# Main Interface
# ===========================
st.title("🏥 Real-time Prediction of In-hospital Mortality for COPD Patients in ICU")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Clinical Profile")
    # Displaying table with English headers
    st.table(input_df.T.rename(columns={0: 'Value'}))

with col2:
    st.subheader("Model Prediction")
    prob = model.predict_proba(input_df)[0][1]

    # Dynamic color mapping
    color = "green" if prob < 0.3 else "orange" if prob < 0.6 else "red"
    st.markdown(f"### Predicted Mortality Risk: <span style='color:{color}'>{prob:.2%}</span>", unsafe_allow_html=True)

    # Risk progress bar
    st.progress(prob)

    if prob > 0.6:
        st.error("⚠️ High Risk: Intensive monitoring and immediate intervention suggested.")
    elif prob > 0.3:
        st.warning("🔔 Moderate Risk: Close observation of clinical status required.")
    else:
        st.success("✅ Low Risk: Patient status appears relatively stable.")

st.markdown("---")

# ===========================
# SHAP Visualization
# ===========================
st.header("📊 Model Interpretation (SHAP Global Interpretation)")
st.write("The following beeswarm plot illustrates the contribution of each feature to the model's prediction of in-hospital mortality:")

plt.close()
fig = plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_v,
    background_data,
    show=False,
    plot_type="dot"
)
current_fig = plt.gcf()
plt.tight_layout()

st.pyplot(current_fig)

st.divider()
st.caption("Note: This tool is based on a Random Forest model developed for research purposes. Clinical decisions should be made in conjunction with professional medical judgment.")