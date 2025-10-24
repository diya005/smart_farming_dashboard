import streamlit as st
import pandas as pd
import joblib

# Load trained models
model_irrigation = joblib.load("models/model_irrigation.pkl")
model_pesticide  = joblib.load("models/model_pesticide.pkl")
model_health     = joblib.load("models/model_health.pkl")
model_yield      = joblib.load("models/model_yield.pkl")



# Streamlit UI
st.title("🌾 Smart Farming AI Advisor")
st.sidebar.header("🌿 Input Field Conditions")

# User inputs
moisture = st.sidebar.slider("Soil Moisture", 0.0, 1.0, 0.2)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 150.0, 20.0)
humidity = st.sidebar.slider("Average Humidity (%)", 10, 100, 60)
mean_temp = st.sidebar.slider("Mean Temperature (°C)", 5, 45, 28)
min_temp = st.sidebar.slider("Min Temperature (°C)", 0, 40, 20)
max_temp = st.sidebar.slider("Max Temperature (°C)", 10, 55, 35)
alkaline = st.sidebar.selectbox("Is soil alkaline?", [0, 1])
sandy = st.sidebar.selectbox("Is soil sandy?", [0, 1])
chalky = st.sidebar.selectbox("Is soil chalky?", [0, 1])
clay = st.sidebar.selectbox("Is soil clay?", [0, 1])



# Input DataFrame
input_df = pd.DataFrame([{
    "Moisture": moisture,
    "rainfall": rainfall,
    "Average Humidity": humidity,
    "Mean Temp": mean_temp,
    "Min temp": min_temp,
    "max Temp": max_temp,
    "alkaline": alkaline,
    "sandy": sandy,
    "chalky": chalky,
    "clay": clay
}])

st.write("### 📋 Input Summary")
st.dataframe(input_df)

# === Feature-specific inputs ===
X_i = input_df[["Moisture", "rainfall", "Average Humidity", "Mean Temp"]]
X_p = input_df[["Moisture", "rainfall", "Average Humidity", "Mean Temp", "alkaline", "sandy", "clay"]]
X_h = input_df[["Moisture", "Average Humidity", "Mean Temp", "alkaline", "sandy", "clay"]]

# Predict intermediate values
pred_irrigation = model_irrigation.predict(X_i)[0]
pred_pesticide = model_pesticide.predict(X_p)[0]
pred_health = model_health.predict(X_h)[0]

# Prepare final input for yield model
input_df["pesticide_dose"] = pred_pesticide
input_df["crop_health_score"] = pred_health

X_y = input_df[["Moisture", "rainfall", "Average Humidity", "Mean Temp", "Min temp", "max Temp",
                "alkaline", "sandy", "chalky", "clay", "pesticide_dose", "crop_health_score"]]
st.text(f"DEBUG: Inputs to Yield Model: {X_y.iloc[0].to_dict()}")

pred_yield = max(0, model_yield.predict(X_y)[0])


# === Results Display ===
st.markdown("---")
st.subheader("🤖 AI Predictions")
st.success(f"💧 Irrigation Needed: {'Yes' if pred_irrigation == 1 else 'No'}")
st.info(f"🧪 Pesticide Dose: {pred_pesticide:.2f} ml/hectare")
st.success(f"🌿 Crop Health Score: {pred_health:.2f}")
st.success(f"🌾 Predicted Yield: {pred_yield:.2f} kg/hectare")

