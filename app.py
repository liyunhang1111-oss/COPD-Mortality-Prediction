# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. 页面基本配置
st.set_page_config(page_title="AECOPD Risk Assessment", layout="wide")

# 2. 静态资源加载 (加了报错保护)
@st.cache_resource
def load_all_res():
    try:
        model = joblib.load('model_CatBoost_size13.pkl')
        # 特征顺序定义
        f_names = [
            'sapsii', 'lab_24hour_firstrr', 'lab_24hour_firsthr', 'first_ptt',
            'first_urea_nitrogen', 'lab_24hour_firsttemperaturef', 'first_platelet_count',
            'first_lactate', 'first_glucose', 'lab_24hour_firstspo2',
            'first_white_blood_cells', 'first_rdw', 'first_po2'
        ]
        # SHAP 解释器初始化
        if hasattr(model, 'calibrated_classifiers_'):
            explainer = shap.TreeExplainer(model.calibrated_classifiers_[0].estimator)
        else:
            explainer = shap.TreeExplainer(model)
        return model, f_names, explainer
    except Exception as e:
        st.error(f"Resource Load Error: {e}")
        return None, None, None

model, feature_names, explainer = load_all_res()

# 名称显示映射
name_map = {
    'sapsii': 'SAPS II Score', 'lab_24hour_firstrr': 'Resp Rate',
    'lab_24hour_firsthr': 'Heart Rate', 'first_ptt': 'PTT (sec)',
    'first_urea_nitrogen': 'BUN (mg/dL)', 'lab_24hour_firsttemperaturef': 'Temp (°F)',
    'first_platelet_count': 'Platelets', 'first_lactate': 'Lactate',
    'first_glucose': 'Glucose', 'lab_24hour_firstspo2': 'SpO2 (%)',
    'first_white_blood_cells': 'WBC', 'first_rdw': 'RDW (%)', 'first_po2': 'PO2 (mmHg)'
}

# 3. 侧边栏表单 (修正了 Duplicate ID 问题)
st.sidebar.header("📋 Clinical Input")

# 使用 unique_key 确保表单 ID 唯一
with st.sidebar.form(key="copd_prediction_form_v1"):
    st.subheader("Physiological Indicators")
    c1, c2 = st.columns(2)
    with c1:
        v1 = st.number_input("SAPS II Score", 0, 150, 40)
        v2 = st.number_input("Resp Rate (bpm)", 0, 100, 20)
        v3 = st.number_input("Heart Rate (bpm)", 0, 250, 85)
        v4 = st.number_input("PTT (sec)", 0, 200, 30)
        v5 = st.number_input("BUN (mg/dL)", 0.0, 200.0, 25.0)
        v6 = st.number_input("Temp (°F)", 70.0, 110.0, 98.6)
        v7 = st.number_input("Platelets", 0, 1000, 200)
    with c2:
        v8 = st.number_input("Lactate (mmol/L)", 0.0, 30.0, 2.0)
        v9 = st.number_input("Glucose", 0, 1000, 120)
        v10 = st.number_input("SpO2 (%)", 0, 100, 95)
        v11 = st.number_input("WBC", 0.0, 100.0, 10.0)
        v12 = st.number_input("RDW (%)", 0.0, 30.0, 14.5)
        v13 = st.number_input("PO2 (mmHg)", 0, 800, 80)
    
    submit = st.form_submit_button("🚀 Run Assessment")

# 4. 主界面展示
st.title("🏥 AECOPD In-hospital Mortality Risk Assessment")
st.markdown("---")

if submit and model:
    # 构造输入
    vals = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13]
    df = pd.DataFrame([vals], columns=feature_names)
    
    col_l, col_r = st.columns([1, 1.2])
    
    with col_l:
        st.subheader("Patient Profile")
        st.table(df.T.rename(index=name_map).rename(columns={0: "Value"}))

    with col_r:
        st.subheader("Model Prediction")
        # 强制通知用户正在计算
        st.write(f"🔄 Calculating risk for SAPS II = {v1}...")
        
        prob = model.predict_proba(df)[0][1]
        color = "red" if prob > 0.5 else "orange" if prob > 0.2 else "green"
        st.markdown(f"## Risk: <span style='color:{color}'>{prob:.2%}</span>", unsafe_allow_html=True)
        st.progress(prob)
        
        # 5. SHAP 解释
        st.markdown("---")
        st.subheader("📊 Feature Contribution (SHAP)")
        try:
            plt.clf()
            sh_val = explainer(df)
            sh_val.feature_names = [name_map.get(n, n) for n in feature_names]
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(sh_val[0], show=False)
            st.pyplot(fig)
        except:
            st.warning("Visualization failed, but prediction is complete.")
else:
    st.info("👈 Please adjust values in the sidebar and click the button to start.")









