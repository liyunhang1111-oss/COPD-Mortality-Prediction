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

# 隐藏默认菜单
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
        
        # 处理校准模型
        if hasattr(model, 'calibrated_classifiers_'):
            base_model = model.calibrated_classifiers_[0].estimator
            explainer = shap.TreeExplainer(base_model)
        else:
            explainer = shap.TreeExplainer(model)
            
        return model, feature_names, explainer
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

model, feature_names, explainer = load_resources()

# 显示名称映射
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
# 3. 侧边栏：13个核心特征输入 (使用 Form 增强稳健性)
# ===========================
st.sidebar.header("📋 Patient Clinical Data")

with st.sidebar.form("input_form"):
    st.subheader("Physiological & Lab Indicators")
    c1, c2 = st.columns(2)
    
    with c1:
        sapsii = st.number_input("SAPS II Score", 0, 150, 40, key="saps")
        rr = st.number_input("Resp Rate (bpm)", 0, 100, 20, key="rr")
        hr = st.number_input("Heart Rate (bpm)", 0, 250, 85, key="hr")
        ptt = st.number_input("PTT (sec)", 0, 200, 30, key="ptt")
        bun = st.number_input("BUN (mg/dL)", 0.0, 200.0, 25.0, key="bun")
        temp = st.number_input("Temp (°F)", 70.0, 110.0, 98.6, key="temp")
        platelet = st.number_input("Platelets", 0, 1000, 200, key="plt")

    with c2:
        lactate = st.number_input("Lactate (mmol/L)", 0.0, 30.0, 2.0, key="lac")
        glucose = st.number_input("Glucose", 0, 1000, 120, key="glu")
        spo2 = st.number_input("SpO2 (%)", 0, 100, 95, key="spo2")
        wbc = st.number_input("WBC", 0.0, 100.0, 10.0, key="wbc")
        rdw = st.number_input("RDW (%)", 0.0, 30.0, 14.5, key="rdw")
        po2 = st.sidebar.number_input("PO2 (mmHg)", 0, 800, 80, key="po2")

    # 提交按钮：强制触发模型预测
    submit_button = st.form_submit_button(label='🚀 Run Assessment')# -*- coding: utf-8 -*-
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

# 隐藏默认菜单
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
        # 确保文件名与 GitHub 上的 model_CatBoost_size13.pkl 完全一致
        model = joblib.load('model_CatBoost_size13.pkl')
        feature_names = [
            'sapsii', 'lab_24hour_firstrr', 'lab_24hour_firsthr', 'first_ptt',
            'first_urea_nitrogen', 'lab_24hour_firsttemperaturef', 'first_platelet_count',
            'first_lactate', 'first_glucose', 'lab_24hour_firstspo2',
            'first_white_blood_cells', 'first_rdw', 'first_po2'
        ]
        
        # 处理校准模型嵌套问题
        if hasattr(model, 'calibrated_classifiers_'):
            base_model = model.calibrated_classifiers_[0].estimator
            explainer = shap.TreeExplainer(base_model)
        else:
            explainer = shap.TreeExplainer(model)
            
        return model, feature_names, explainer
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

model, feature_names, explainer = load_resources()

# 显示名称映射
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
st.sidebar.header("📋 Patient Clinical Data")

# 使用表单模式强制刷新
with st.sidebar.form("input_form"):
    st.subheader("Physiological & Lab Indicators")
    c1, c2 = st.columns(2)
    
    with c1:
        s_sapsii = st.number_input("SAPS II Score", 0, 150, 40)
        s_rr = st.number_input("Resp Rate (bpm)", 0, 100, 20)
        s_hr = st.number_input("Heart Rate (bpm)", 0, 250, 85)
        s_ptt = st.number_input("PTT (sec)", 0, 200, 30)
        s_bun = st.number_input("BUN (mg/dL)", 0.0, 200.0, 25.0)
        s_temp = st.number_input("Temp (°F)", 70.0, 110.0, 98.6)
        s_platelet = st.number_input("Platelets", 0, 1000, 200)

    with c2:
        s_lactate = st.number_input("Lactate (mmol/L)", 0.0, 30.0, 2.0)
        s_glucose = st.number_input("Glucose", 0, 1000, 120)
        s_spo2 = st.number_input("SpO2 (%)", 0, 100, 95)
        s_wbc = st.number_input("WBC", 0.0, 100.0, 10.0)
        s_rdw = st.number_input("RDW (%)", 0.0, 30.0, 14.5)
        s_po2 = st.number_input("PO2 (mmHg)", 0, 800, 80)

    # 提交按钮
    submit_button = st.form_submit_button(label='🚀 Run Assessment')

# ===========================
# 4. 主界面：结果展示
# ===========================
st.title("🏥 In-hospital Mortality Risk Assessment for COPD Patients in ICU")
st.write("A Clinical Decision Support Tool based on Multi-center Big Data and CatBoost.")
st.markdown("---")

# 只有点击按钮后才执行计算逻辑
if submit_button:
    # 1. 构建实时输入的数据框
    data = {
        'sapsii': s_sapsii, 'lab_24hour_firstrr': s_rr, 'lab_24hour_firsthr': s_hr,
        'first_ptt': s_ptt, 'first_urea_nitrogen': s_bun, 'lab_24hour_firsttemperaturef': s_temp,
        'first_platelet_count': s_platelet, 'first_lactate': s_lactate, 'first_glucose': s_glucose,
        'lab_24hour_firstspo2': s_spo2, 'first_white_blood_cells': s_wbc, 'first_rdw': s_rdw, 'first_po2': s_po2
    }
    input_df = pd.DataFrame(data, index=[0])[feature_names]

    # 2. 布局显示
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Patient Clinical Profile")
        display_df = input_df.T.rename(index=name_mapping)
        display_df.columns = ['Value']
        st.table(display_df)

    with col2:
        st.subheader("Model Prediction")
        
        # 调试输出：确认后台收到的数值（如果它变了，概率就一定会变）
        st.write(f"🔄 Processing Input... (SAPS II: {s_sapsii})")
        
        # 核心：执行预测
        prob = model.predict_proba(input_df)[0][1]
        
        color = "green" if prob < 0.2 else "orange" if prob < 0.5 else "red"
        st.markdown(f"### Predicted Mortality Risk: <span style='color:{color}'>{prob:.2%}</span>", unsafe_allow_html=True)
        st.progress(prob)

        if prob > 0.5:
            st.error("⚠️ **High Risk**: Enhanced clinical monitoring suggested.")
        elif prob > 0.2:
            st.warning("🔔 **Moderate Risk**: Close observation required.")
        else:
            st.success("✅ **Low Risk**: Clinically stable profile.")
        st.caption("Probability is post-hoc calibrated via Platt Scaling.")

    st.markdown("---")

    # 3. SHAP 解释
    st.header("📊 Individual Risk Interpretation (SHAP Waterfall)")
    try:
        plt.clf()
        shap_input = input_df.copy()
        shap_input.columns = [name_mapping.get(col, col) for col in shap_input.columns]
        
        shap_values = explainer(input_df)
        shap_values.feature_names = list(shap_input.columns)
        shap_values.data = shap_input.values
        
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"SHAP Error: {e}")
else:
    # 默认提示
    st.info("👈 Please adjust clinical parameters in the sidebar and click 'Run Assessment' to see the predicted results.")

st.markdown("---")
st.caption("Disclaimer: This tool is for research purposes only and should not replace clinical judgment.")








