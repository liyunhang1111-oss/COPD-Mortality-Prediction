# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. 页面基本配置
st.set_page_config(page_title="AECOPD Risk Assessment", layout="wide")

# 2. 资源加载 (移除缓存，强制实时)
def load_model_and_names():
    try:
        model = joblib.load('model_CatBoost_size13.pkl')
        f_names = [
            'sapsii', 'lab_24hour_firstrr', 'lab_24hour_firsthr', 'first_ptt',
            'first_urea_nitrogen', 'lab_24hour_firsttemperaturef', 'first_platelet_count',
            'first_lactate', 'first_glucose', 'lab_24hour_firstspo2',
            'first_white_blood_cells', 'first_rdw', 'first_po2'
        ]
        return model, f_names
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None, None

model, feature_names = load_model_and_names()

# 名称显示映射
name_map = {
    'sapsii': 'SAPS II Score', 'lab_24hour_firstrr': 'Resp Rate',
    'lab_24hour_firsthr': 'Heart Rate', 'first_ptt': 'PTT (sec)',
    'first_urea_nitrogen': 'BUN (mg/dL)', 'lab_24hour_firsttemperaturef': 'Temp (°F)',
    'first_platelet_count': 'Platelets', 'first_lactate': 'Lactate',
    'first_glucose': 'Glucose', 'lab_24hour_firstspo2': 'SpO2 (%)',
    'first_white_blood_cells': 'WBC', 'first_rdw': 'RDW (%)', 'first_po2': 'PO2 (mmHg)'
}

# 3. 侧边栏表单
st.sidebar.header("📋 Clinical Input")
with st.sidebar.form(key="final_form_v2"):
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
    # 构造输入矩阵
    vals = np.array([[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13]])
    df = pd.DataFrame(vals, columns=feature_names)
    
    col_l, col_r = st.columns([1, 1.2])
    
    with col_l:
        st.subheader("Patient Profile")
        st.table(df.T.rename(index=name_map).rename(columns={0: "Value"}))

    with col_r:
        st.subheader("Model Prediction")
        
        # --- 核心修改：强制重新计算概率 ---
        prob_array = model.predict_proba(df)
        prob = float(prob_array[0][1]) # 强制转换为浮点数
        
        color = "red" if prob > 0.5 else "orange" if prob > 0.2 else "green"
        st.markdown(f"## Risk: <span style='color:{color}'>{prob:.2%}</span>", unsafe_allow_html=True)
        st.progress(prob)
        
        # --- 核心修改：强制重新计算 SHAP ---
        st.markdown("---")
        st.subheader("📊 Feature Contribution (SHAP)")
        try:
            # 每次点击按钮都重新定义解释器，防止缓存
            if hasattr(model, 'calibrated_classifiers_'):
                inner_model = model.calibrated_classifiers_[0].estimator
            else:
                inner_model = model
            
            explainer = shap.TreeExplainer(inner_model)
            shap_values = explainer(df)
            
            # 替换特征名显示
            shap_values.feature_names = [name_map.get(n, n) for n in feature_names]
            
            plt.clf()
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP 计算失败: {e}")
else:
    st.info("👈 请在左侧输入参数并点击 'Run Assessment'")










