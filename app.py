import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Wine Classifier 🍷",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CUSTOM CSS =====================
st.markdown("""
    <style>
    .main {
        background-color: #fdf6f0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #7b1e3b, #a52a4c);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.6em 1.5em;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #a52a4c, #7b1e3b);
        transform: scale(1.03);
    }
    .title-text {
        text-align: center;
        font-size: 2.6em;
        font-weight: 800;
        color: #6b1d33;
    }
    .subtitle-text {
        text-align: center;
        color: #5a5a5a;
        font-size: 1.1em;
        margin-bottom: 25px;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        margin-top: 15px;
    }
    .footer {
        text-align: center;
        color: gray;
        margin-top: 50px;
        font-size: 0.85em;
    }
    </style>
""", unsafe_allow_html=True)

# ===================== LOAD MODEL & DATA =====================
model = joblib.load("wine_model.joblib")
wine = load_wine()

# Color theme per class
class_colors = {
    0: "#8e44ad",  # purple
    1: "#27ae60",  # green
    2: "#c0392b"   # red
}
class_emojis = {0: "🍇", 1: "🍏", 2: "🍷"}

# ===================== SIDEBAR =====================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/135/135692.png", width=90)
    st.title("About this App")
    st.markdown("""
    This app predicts the **cultivar class** of wine 
    based on its chemical composition using a trained 
    **Machine Learning model**.

    **Dataset:** UCI Wine Dataset  
    **Classes:**
    - 🍇 Class 0  
    - 🍏 Class 1  
    - 🍷 Class 2  
    """)
    st.markdown("---")
    st.subheader("⚙️ Input Guide")
    st.caption("Adjust the sliders to match the wine's chemical profile, then click **Predict**.")
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")

# ===================== HEADER =====================
st.markdown('<div class="title-text">🍷 Wine Classification App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Predict the wine cultivar class from its chemical attributes</div>', unsafe_allow_html=True)

st.markdown("---")

# ===================== TABS =====================
tab1, tab2 = st.tabs(["🔮 Predict", "📊 Feature Info"])

# ===================== TAB 1: PREDICTION =====================
with tab1:
    st.subheader("Enter Wine Chemical Features")

    col1, col2 = st.columns(2)

    with col1:
        alcohol = st.slider("🍾 Alcohol", 11.0, 15.0, 13.0, 0.1)
        malic_acid = st.slider("🍋 Malic Acid", 0.7, 5.8, 2.3, 0.1)
        ash = st.slider("🧂 Ash", 1.3, 3.2, 2.3, 0.1)

    with col2:
        alcalinity = st.slider("⚗️ Alcalinity of Ash", 10.6, 30.0, 19.5, 0.1)
        magnesium = st.slider("🧲 Magnesium", 70.0, 162.0, 99.0, 1.0)

    st.markdown("### 📋 Summary of Inputs")
    input_df = pd.DataFrame({
        "Feature": ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium"],
        "Value": [alcohol, malic_acid, ash, alcalinity, magnesium]
    })
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    predict_btn = st.button("🔮 Predict Wine Class", use_container_width=True)

    if predict_btn:
        with st.spinner("Analyzing chemical composition..."):
            features = [alcohol, malic_acid, ash, alcalinity, magnesium] + [0.0] * 8
            prediction = model.predict([features])[0]
            target_name = wine.target_names[prediction]
            color = class_colors[prediction]
            emoji = class_emojis[prediction]

        st.markdown(
            f"""
            <div class="result-box" style="background-color:{color}20; border: 2px solid {color}; color:{color};">
                {emoji} Predicted Class: {target_name.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Try to show probabilities if supported
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([features])[0]
            st.markdown("#### 📈 Prediction Confidence")
            proba_df = pd.DataFrame({
                "Class": [f"{class_emojis[i]} {wine.target_names[i]}" for i in range(len(proba))],
                "Probability": proba
            })
            st.bar_chart(proba_df.set_index("Class"))

            for i, p in enumerate(proba):
                st.progress(float(p), text=f"{class_emojis[i]} {wine.target_names[i]}: {p*100:.1f}%")

        st.balloons()

# ===================== TAB 2: FEATURE INFO =====================
with tab2:
    st.subheader("📖 Understanding the Features")
    st.markdown("""
    | Feature | Description |
    |----------|-------------|
    | **Alcohol** | Alcohol content of the wine |
    | **Malic Acid** | Acid contributing to tartness |
    | **Ash** | Inorganic residue after burning |
    | **Alcalinity of Ash** | Alkalinity level of the ash |
    | **Magnesium** | Mineral content in the wine |
    """)
    st.info("Note: The model also uses 8 additional chemical features internally, "
            "set to default values in this simplified interface.")

    st.markdown("### 🔬 Dataset Snapshot")
    df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
    df_wine["target"] = [wine.target_names[t] for t in wine.target]
    st.dataframe(df_wine.head(10), use_container_width=True)

# ===================== FOOTER =====================
st.markdown('<div class="footer">© 2024 Wine Classifier App | Powered by Streamlit & Scikit-learn</div>', unsafe_allow_html=True)