# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# ===========================
# 1. 页面配置与专业文案
# ===========================
st.set_page_config(
    page_title="COPD ICU Mortality Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏冗余菜单
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ===========================
# 2. 资源加载 (核心模型与 SHAP)
# ===========================
@st.cache_resource
def load_resources():
    try:
        # 加载您刚刚上传的新版扁平化校准模型
        model = joblib.load('model_CatBoost_calibrated_final.pkl')
        
        # 13个核心特征名称（必须与训练时严格一致）
        f_names = [
            'sapsii', 'lab_24hour_firstrr', 'lab_24hour_firsthr', 'first_ptt',
            'first_urea_nitrogen', 'lab_24hour_firsttemperaturef', 'first_platelet_count',
            'first_lactate', 'first_glucose', 'lab_24hour_firstspo2',
            'first_white_blood_cells', 'first_rdw', 'first_po2'
        ]
        
        # 提取校准器内部的基础 CatBoost 模型用于 SHAP 解释
        # 这一步是解决“图表不动”的关键
        if hasattr(model, 'calibrated_classifiers_'):
            inner_model = model.calibrated_classifiers_[0].estimator
        else:
            inner_model = model
            
        explainer = shap.TreeExplainer(inner_model)
        return model, f_names, explainer
    except Exception as e:
        st.error(f"Resource Load Error: {e}")
        return None, None, None

model, feature_names, explainer = load_resources()

# 临床指标名称映射
name_map = {
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
# 3. 侧边栏：临床数据录入
# ===========================
st.sidebar.header("📋 Patient Clinical Data")

with st.sidebar.form(key="icu_copd_form"):
    st.subheader("Physiological & Lab Results")
    c1, c2 = st.columns(2)
    with c1:
        v1 = st.number_input("SAPS II Score", 0, 150, 44)
        v2 = st.number_input("Resp Rate (bpm)", 0, 100, 22)
        v3 = st.number_input("Heart Rate (bpm)", 0, 250, 88)
        v4 = st.number_input("PTT (sec)", 0, 200, 32)
        v5 = st.number_input("BUN (mg/dL)", 0.0, 250.0, 28.0)
        v6 = st.number_input("Temp (°F)", 70.0, 110.0, 98.4)
        v7 = st.number_input("Platelets (10³/µL)", 0, 1000, 180)
    with c2:
        v8 = st.number_input("Lactate (mmol/L)", 0.0, 30.0, 2.1)
        v9 = st.number_input("Glucose (mg/dL)", 0, 1000, 130)
        v10 = st.number_input("SpO2 (%)", 0, 100, 94)
        v11 = st.number_input("WBC (10³/µL)", 0.0, 100.0, 11.2)
        v12 = st.number_input("RDW (%)", 0.0, 35.0, 15.1)
        v13 = st.number_input("PO2 (mmHg)", 0, 800, 75)
    
    submit_button = st.form_submit_button(label='🚀 Run Assessment')

# ===========================
# 4. 主界面展示
# ===========================
st.title("🏥 In-hospital Mortality Risk Assessment for COPD Patients in ICU")
st.markdown("""
    This clinical decision support tool utilizes a Calibrated CatBoost model to predict the in-hospital mortality risk for COPD patients in the ICU, based on the first measured values of clinical variables upon ICU admission.
    """)
st.markdown("---")

if submit_button and model:
    # 构造实时输入数据
    input_vals = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13]
    input_df = pd.DataFrame([input_vals], columns=feature_names)
    
    # 执行实时预测
    # 由于使用了 cv='prefit'，此处得到的概率是经过 Platt Scaling 校准的
    prob = model.predict_proba(input_df)[0][1]
    
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("Patient Clinical Profile")
        # 转换为表格展示
        display_df = input_df.T.rename(index=name_map)
        display_df.columns = ['Value']
        st.table(display_df)

    with col_right:
        st.subheader("Risk Prediction")
        
        # 调试信息：确保每次计算时前端都会刷新显示当前值
        st.write(f"🔄 **Computing risk for SAPS II = {v1}...**")
        
        # 动态配色方案
        color = "red" if prob > 0.5 else "orange" if prob > 0.2 else "green"
        st.markdown(f"### Predicted Probability: <span style='color:{color}'>{prob:.2%}</span>", unsafe_allow_html=True)
        st.progress(prob)
        
        # 临床解释
        if prob > 0.5:
            st.error("⚠️ **High Risk Group**: Intensive monitoring and aggressive intervention recommended.")
        elif prob > 0.1:
            st.warning("🔔 **Moderate Risk Group**: Close clinical observation required.")
        else:
            st.success("✅ **Low Risk Group**: Standard ICU clinical protocol suggested.")
        
        st.caption("Probability calibrated via Platt Scaling. Model based on MIMIC-IV and eICU cohorts.")

    st.markdown("---")

    # 5. SHAP 解释
    st.subheader("📊 Individual Risk Factors Interpretation (SHAP Waterfall)")
    try:
        # 重置绘图区域
        plt.clf()
        
        # 计算当前输入的 SHAP 值
        shap_values = explainer(input_df)
        
        # 将内部特征名替换为临床可读名称
        shap_values.feature_names = [name_map.get(n, n) for n in feature_names]
        
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("The waterfall plot shows how each clinical factor contributes to the deviation from the average risk.")
    except Exception as e:
        st.error(f"SHAP interpretation failed: {e}")

else:
    st.info("👈 Please enter patient data in the sidebar and click **'Run Assessment'** to generate results.")

st.markdown("---")
st.caption("Disclaimer: This tool is intended for research and educational purposes only. It should not be used as a substitute for professional medical judgment.")











