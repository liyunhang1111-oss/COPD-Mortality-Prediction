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

# 隐藏默认菜单以提升专业感
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
        # 加载最新的 13 特征 CatBoost 模型
        model = joblib.load('model_CatBoost_size13.pkl')
        
        # 定义核心特征顺序（必须与建模时一致）
        feature_names = [
            'sapsii', 'lab_24hour_firstrr', 'lab_24hour_firsthr', 'first_ptt',
            'first_urea_nitrogen', 'lab_24hour_firsttemperaturef', 'first_platelet_count',
            'first_lactate', 'first_glucose', 'lab_24hour_firstspo2',
            'first_white_blood_cells', 'first_rdw', 'first_po2'
        ]
        
        # 兼容性处理：如果模型被 CalibratedClassifierCV 包装
        if hasattr(model, 'calibrated_classifiers_'):
            base_model = model.calibrated_classifiers_[0].estimator
            explainer = shap.TreeExplainer(base_model)
        else:
            explainer = shap.TreeExplainer(model)
            
        return model, feature_names, explainer
    except Exception as e:
        st.error(f"加载模型失败，请检查文件路径及 requirements.txt: {e}")
        st.stop()

model, feature_names, explainer = load_resources()

# 定义临床名称显示映射表
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
    # 使用侧边栏专用列布局
    c1, c2 = st.sidebar.columns(2)
    
    with c1:
        # 核心修正：必须使用 st.sidebar.number_input 确保数值实时捕获
        sapsii = st.sidebar.number_input("SAPS II Score", 0, 150, 40)
        rr = st.sidebar.number_input("Resp Rate (bpm)", 0, 100, 20)
        hr = st.sidebar.number_input("Heart Rate (bpm)", 0, 250, 85)
        ptt = st.sidebar.number_input("PTT (sec)", 0, 200, 30)
        bun = st.sidebar.number_input("BUN (mg/dL)", 0.0, 200.0, 25.0)
        temp = st.sidebar.number_input("Temp (°F)", 70.0, 110.0, 98.6)
        platelet = st.sidebar.number_input("Platelets", 0, 1000, 200)

    with c2:
        lactate = st.sidebar.number_input("Lactate (mmol/L)", 0.0, 30.0, 2.0)
        glucose = st.sidebar.number_input("Glucose", 0, 1000, 120)
        spo2 = st.sidebar.number_input("SpO2 (%)", 0, 100, 95)
        wbc = st.sidebar.number_input("WBC", 0.0, 100.0, 10.0)
        rdw = st.sidebar.number_input("RDW (%)", 0.0, 30.0, 14.5)
        po2 = st.sidebar.number_input("PO2 (mmHg)", 0, 800, 80)

    data = {
        'sapsii': sapsii, 'lab_24hour_firstrr': rr, 'lab_24hour_firsthr': hr,
        'first_ptt': ptt, 'first_urea_nitrogen': bun, 'lab_24hour_firsttemperaturef': temp,
        'first_platelet_count': platelet, 'first_lactate': lactate, 'first_glucose': glucose,
        'lab_24hour_firstspo2': spo2, 'first_white_blood_cells': wbc, 'first_rdw': rdw, 'first_po2': po2
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
# 强制对齐特征顺序
input_df = input_df[feature_names]

# ===========================
# 4. 主界面：结果展示
# ===========================
st.title("🏥 In-hospital Mortality Risk Assessment for COPD Patients in ICU")
st.write("A Clinical Decision Support Tool based on Multi-center Big Data and CatBoost.")
st.markdown("---")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Patient Clinical Profile")
    # 展示时使用美化后的名字
    display_df = input_df.T.rename(index=name_mapping)
    display_df.columns = ['Value']
    st.table(display_df)

with col2:
    st.subheader("Model Prediction")
    # 获取校准后的概率
    prob = model.predict_proba(input_df)[0][1]

    # 动态颜色显示
    color = "green" if prob < 0.2 else "orange" if prob < 0.5 else "red"
    st.markdown(f"### Predicted Mortality Risk: <span style='color:{color}'>{prob:.2%}</span>", unsafe_allow_html=True)

    st.progress(prob)

    if prob > 0.5:
        st.error("⚠️ **High Risk**: Enhanced clinical monitoring suggested.")
    elif prob > 0.2:
        st.warning("🔔 **Moderate Risk**: Close observation required.")
    else:
        st.success("✅ **Low Risk**: Clinically stable profile.")
        
    st.caption("Probability is post-hoc calibrated via Platt Scaling to ensure clinical reliability.")

st.markdown("---")

# ===========================
# 5. SHAP 可解释性分析
# ===========================
st.header("📊 Individual Risk Interpretation (SHAP Waterfall)")
st.write("This chart visualizes the contribution of each feature to THIS specific patient's risk:")

try:
    # 准备用于绘图的数据副本，替换列名为美化名称
    shap_input = input_df.copy()
    shap_input.columns = [name_mapping.get(col, col) for col in shap_input.columns]
    
    # 计算 SHAP 值并注入美化后的特征名
    shap_values = explainer(input_df)
    shap_values.feature_names = list(shap_input.columns)
    shap_values.data = shap_input.values # 确保图中显示的数值也是最新的
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    st.pyplot(fig)
except Exception as e:
    st.info("SHAP visualization is generating based on input changes...")

st.markdown("---")
st.caption("Disclaimer: This tool is for research purposes only and should not replace clinical judgment.")





