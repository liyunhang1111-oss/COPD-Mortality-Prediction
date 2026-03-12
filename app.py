# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# ===========================
# 1. 页面配置与美化
# ===========================
st.set_page_config(
    page_title="COPD Mortality Prediction (ICU)",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_style, unsafe_allow_html=True)

# ===========================
# 2. 资源加载 (模型与 SHAP)
# ===========================
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model_CatBoost_size13.pkl')
        
        feature_names = [
            'sapsii', 'lab_24hour_firstrr', 'lab_24hour_firsthr', 'first_ptt',
            'first_urea_nitrogen', 'lab_24hour_firsttemperaturef', 'first_platelet_count',
            'first_lactate', 'first_glucose', 'lab_24hour_firstspo2',
            'first_white_blood_cells', 'first_rdw', 'first_po2'
        ]
        
        if hasattr(model, 'calibrated_classifiers_'):
            base_model = model.calibrated_classifiers_[0].estimator
            explainer = shap.TreeExplainer(base_model)
        else:
            explainer = shap.TreeExplainer(model)
            
        return model, feature_names, explainer
    except Exception as e:
        st.error(f"加载模型失败，请检查文件路径: {e}")
        st.stop()

model, feature_names, explainer = load_resources()

# --- 新增：定义显示名称映射字典 ---
name_mapping = {
    'sapsii': 'SAPS II Score',
    'lab_24hour_firstrr': 'Respiratory Rate',
    'lab_24hour_firsthr': 'Heart Rate',
    'first_ptt': 'PTT (sec)',
    'first_urea_nitrogen': 'BUN (mg/dL)',
    'lab_24hour_firsttemperaturef': 'Temperature (°F)',
    'first_platelet_count': 'Platelets',
    'first_lactate': 'Lactate',
    'first_glucose': 'Glucose',
    'lab_24hour_firstspo2': 'SpO2 (%)',
    'first_white_blood_cells': 'WBC',
    'first_rdw': 'RDW (%)',
    'first_po2': 'PO2 (mmHg)'
}

# ===========================
# 3. 侧边栏：13个核心特征输入
# ===========================
st.sidebar.header("📋 Clinical Parameters")

def user_input_features():
    st.sidebar.subheader("Physiological & Lab Indicators")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        sapsii = st.number_input("SAPS II Score", 0, 150, 40)
        rr = st.number_input("Resp Rate (bpm)", 0, 100, 20)
        hr = st.number_input("Heart Rate (bpm)", 0, 250, 85)
        ptt = st.number_input("PTT (sec)", 0, 200, 30)
        bun = st.number_input("BUN (mg/dL)", 0.0, 200.0, 25.0)
        temp = st.number_input("Temp (°F)", 70.0, 110.0, 98.6)
        platelet = st.number_input("Platelets", 0, 1000, 200)

    with c2:
        lactate = st.number_input("Lactate (mmol/L)", 0.0, 30.0, 2.0)
        glucose = st.number_input("Glucose", 0, 1000, 120)
        spo2 = st.number_input("SpO2 (%)", 0, 100, 95)
        wbc = st.number_input("WBC", 0.0, 100.0, 10.0)
        rdw = st.number_input("RDW (%)", 0.0, 30.0, 14.5)
        po2 = st.number_input("PO2 (mmHg)", 0, 800, 80)

    data = {
        'sapsii': sapsii, 'lab_24hour_firstrr': rr, 'lab_24hour_firsthr': hr,
        'first_ptt': ptt, 'first_urea_nitrogen': bun, 'lab_24hour_firsttemperaturef': temp,
        'first_platelet_count': platelet, 'first_lactate': lactate, 'first_glucose': glucose,
        'lab_24hour_firstspo2': spo2, 'first_white_blood_cells': wbc, 'first_rdw': rdw, 'first_po2': po2
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
input_df = input_df[feature_names]

# ===========================
# 4. 主界面：结果展示
# ===========================
st.title("🏥 In-hospital Mortality Risk Assessment for COPD Patients in ICU")
st.write("A Clinical Decision Support Tool for Patients with COPD Admitted to Intensive Care Units.")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Patient Clinical Profile")
    # --- 修改处：在显示时进行重命名，但不改动原始 input_df ---
    display_df = input_df.T.rename(index=name_mapping)
    display_df.columns = ['Value']
    st.table(display_df) # 使用 table 或 dataframe 均可

with col2:
    st.subheader("Model Prediction")
    prob = model.predict_proba(input_df)[0][1]

    color = "green" if prob < 0.2 else "orange" if prob < 0.5 else "red"
    st.markdown(f"### Predicted Mortality Risk: <span style='color:{color}'>{prob:.2%}</span>", unsafe_allow_html=True)

    st.progress(prob)

    if prob > 0.5:
        st.error("⚠️ **High Risk**: Enhanced monitoring suggested.")
    elif prob > 0.2:
        st.warning("🔔 **Moderate Risk**: Close clinical observation required.")
    else:
        st.success("✅ **Low Risk**: Clinically stable.")
        
    st.caption("Probability calibrated via Platt Scaling to ensure clinical reliability.")

st.markdown("---")

# ===========================
# 5. SHAP 局部解释
# ===========================
st.header("📊 Individual Risk Interpretation (SHAP Waterfall)")
st.write("This chart shows how each feature pushes the risk higher (red) or lower (blue) for THIS specific patient:")

try:
    # 为了让 SHAP 图也显示美化后的名字，临时修改副本的列名
    shap_input = input_df.copy()
    shap_input.columns = [name_mapping.get(col, col) for col in shap_input.columns]
    
    # 重新初始化一个带有美化名称的解释器数据
    shap_values = explainer(input_df)
    # 将美化后的名字赋给 SHAP 值对象用于绘图
    shap_values.feature_names = list(shap_input.columns)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())
except Exception as e:
    st.write(f"Note: SHAP individual visualization error: {e}")

st.markdown("---")
st.caption("Disclaimer: For research purposes only. Not for direct clinical diagnosis.")




