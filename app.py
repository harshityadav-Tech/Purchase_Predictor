import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Purchase Predictor", layout="wide")

# ==============================
# Custom CSS (Premium Styling)
# ==============================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #00ADB5;
}
.card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Model
# ==============================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ==============================
# Header
# ==============================
st.markdown('<p class="big-title">🛒 Purchase Prediction Dashboard</p>', unsafe_allow_html=True)
st.write("AI-powered prediction of user purchase behavior")

# ==============================
# Layout Columns
# ==============================
col1, col2 = st.columns([1,1])

# ==============================
# Input Section
# ==============================
with col1:
    st.markdown("### 🔧 Input Features")

    interaction = st.slider("Interaction Count", 1, 50, 3)
    hour = st.slider("Hour of Day", 0, 23, 14)
    weekday = st.selectbox("Weekday", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    category = st.number_input("Category ID", value=1000)

    weekday_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
    weekday = weekday_map[weekday]

# ==============================
# Prediction Section
# ==============================
with col2:
    st.markdown("### 📊 Prediction Result")

    if st.button("Predict 🚀"):

        input_data = pd.DataFrame({
            'interaction': [interaction],
            'hour': [hour],
            'weekday': [weekday],
            'categoryid': [category]
        })

        input_scaled = scaler.transform(input_data)

        prob = model.predict_proba(input_scaled)[:,1]
        prediction = (prob > 0.5).astype(int)

        st.markdown("### 🔍 Output")

        if prediction[0] == 1:
            st.success(f"🟢 Likely to Purchase ({prob[0]*100:.2f}%)")
        else:
            st.error(f"🔴 Not Likely to Purchase ({prob[0]*100:.2f}%)")

