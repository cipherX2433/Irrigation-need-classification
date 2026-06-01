import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pathlib

# ==============================================================================
# 1. PAGE SETUP & ADVANCED CUSTOM CSS SYSTEM
# ==============================================================================
st.set_page_config(
    page_title="Smart Irrigation Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Global UI Injection via CSS
st.markdown("""
    <style>
    /* Global App Background and Content Wrapper Alignment */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Styled Metric Cards for Engineered Features */
    .metric-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #2E7D32;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #555555;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1b5e20;
        margin-top: 5px;
    }

    /* Result Banner Customizations */
    .result-banner {
        padding: 25px;
        border-radius: 12px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .banner-high {
        background-color: #fce8e6;
        border: 1px solid #ea4335;
        color: #c5221f;
    }
    .banner-medium {
        background-color: #fef7e0;
        border: 1px solid #fbbc04;
        color: #b06000;
    }
    .banner-low {
        background-color: #e6f4ea;
        border: 1px solid #34a853;
        color: #137333;
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. CACHED ARTIFACT LOADING
# ==============================================================================
@st.cache_resource
def load_machine_learning_artifacts():
    current_dir = pathlib.Path(__file__).parent.resolve()
    pkl_path = current_dir / "irrigation_app.pkl"
    artifacts = joblib.load(pkl_path)
    return (
        artifacts["model"],
        artifacts["encoders"],
        artifacts["selected_features"],
        artifacts["target_map"]
    )


model, encoders, selected_features, target_map = load_machine_learning_artifacts()

# ==============================================================================
# 3. SIDEBAR USER INTERFACE (INPUT CONTROLS)
# ==============================================================================
st.sidebar.header("🚜 Farm Parameters")

with st.sidebar.expander("🌤️ Weather & Environment", expanded=True):
    temperature = st.sidebar.slider("Temperature (°C)", 12.0, 42.0, 28.0)
    humidity = st.sidebar.slider("Humidity (%)", 25.0, 95.0, 60.0)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, value=1200.0)
    wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.5, 20.0, 10.0)
    sunlight_hours = st.sidebar.slider("Sunlight Hours", 4.0, 11.0, 8.0)

with st.sidebar.expander("🪵 Soil & Crop Metrics", expanded=True):
    soil_moisture = st.sidebar.slider("Soil Moisture", 8.0, 65.0, 30.0)
    electrical_conductivity = st.sidebar.slider("Electrical Conductivity", 0.1, 3.5, 1.5)
    previous_irrigation = st.sidebar.number_input("Previous Irrigation (mm)", min_value=0.0, value=60.0)

with st.sidebar.expander("🌱 Field Configuration", expanded=True):
    growth_stage = st.sidebar.selectbox("Crop Growth Stage", list(encoders["Crop_Growth_Stage"].classes_))
    mulching = st.sidebar.selectbox("Mulching Used", list(encoders["Mulching_Used"].classes_))

st.sidebar.header("⚙️ Optimization Controls")
high_recall_threshold = st.sidebar.slider(
    "High Need Sensitivity Threshold",
    min_value=0.1, max_value=0.5, value=0.3, step=0.05,
    help="Lower thresholds increase Recall for 'High' target to avoid underwatering crops."
)

# ==============================================================================
# 4. DATA PROCESSING & FEATURE ENGINEERING
# ==============================================================================
ET_Proxy = (temperature * sunlight_hours * wind_speed) / (humidity + 1)
Water_Stress_Index = ET_Proxy / (soil_moisture + 1)
Rain_Temp_Ratio = rainfall / (temperature + 1)

growth_stage_encoded = encoders["Crop_Growth_Stage"].transform([growth_stage])[0]
mulching_encoded = encoders["Mulching_Used"].transform([mulching])[0]

input_df = pd.DataFrame([{
    "Soil_Moisture": soil_moisture,
    "Water_Stress_Index": Water_Stress_Index,
    "Rainfall_mm": rainfall,
    "Temperature_C": temperature,
    "Wind_Speed_kmh": wind_speed,
    "Rain_Temp_Ratio": Rain_Temp_Ratio,
    "ET_Proxy": ET_Proxy,
    "Humidity": humidity,
    "Previous_Irrigation_mm": previous_irrigation,
    "Electrical_Conductivity": electrical_conductivity,
    "Crop_Growth_Stage": growth_stage_encoded,
    "Mulching_Used": mulching_encoded
}])
input_df = input_df[selected_features]

# ==============================================================================
# 5. MAIN CONTENT FRAME (DASHBOARD ARCHITECTURE)
# ==============================================================================
st.title("🌱 Smart Irrigation Need Prediction")
st.markdown("Use real-time field telemetry and engineered metrics to predict critical crop hydration status.")
st.markdown("---")

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("💡 Calculated Feature Matrix")

    # Custom HTML Layout blocks replacing the default st.metric wrapper
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Evapotranspiration (ET Proxy)</div>
            <div class="metric-value">{ET_Proxy:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Water Stress Index</div>
            <div class="metric-value">{Water_Stress_Index:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Rainfall / Temperature Ratio</div>
            <div class="metric-value">{Rain_Temp_Ratio:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("🔮 Model Inference Engine")

    if st.button("Predict Irrigation Need", type="primary", use_container_width=True):
        prediction_proba = model.predict_proba(input_df)[0]

        if prediction_proba[2] >= high_recall_threshold:
            final_prediction_index = 2
        else:
            final_prediction_index = np.argmax(prediction_proba)

        result = target_map[final_prediction_index]

        # Inject stylized alert container cards depending on classification result
        if "High" in result:
            st.markdown(f"""
                <div class="result-banner banner-high">
                    <h3 style="margin:0; color:#c5221f;">⚠️ ALERT: High Irrigation Needed</h3>
                    <p style="margin:5px 0 0 0;">Soil/Climate telemetry shows extreme water deficit conditions.</p>
                </div>
            """, unsafe_allow_html=True)
        elif "Medium" in result:
            st.markdown(f"""
                <div class="result-banner banner-medium">
                    <h3 style="margin:0; color:#b06000;">⚠️ NOTICE: Medium Irrigation Needed</h3>
                    <p style="margin:5px 0 0 0;">Monitoring values are moderate. Plan regular watering sequence.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-banner banner-low">
                    <h3 style="margin:0; color:#137333;">✅ EXCELLENT: Low/No Irrigation Needed</h3>
                    <p style="margin:5px 0 0 0;">Crop moisture profiles are completely sufficient.</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📊 Confidence Level Metrics")

        prob_df = pd.DataFrame({
            "Irrigation Need Class": ["Low", "Medium", "High"],
            "Probability (%)": prediction_proba * 100
        })

        sub_col1, sub_col2 = st.columns([2, 3], gap="medium")
        with sub_col1:
            st.write("")
            st.dataframe(
                prob_df.style.format({"Probability (%)": "{:.2f}%"})
                .background_gradient(cmap="Greens", subset=["Probability (%)"]),
                hide_index=True,
                use_container_width=True
            )
        with sub_col2:
            st.bar_chart(prob_df.set_index("Irrigation Need Class"), y="Probability (%)", color="#2E7D32")

# ==============================================================================
# 6. APPLICATION FOOTER
# ==============================================================================
st.markdown("---")
st.caption("🤖 Inference Framework: Scikit-Learn VotingClassifier Engine (LightGBM + XGBoost + CatBoost)")
