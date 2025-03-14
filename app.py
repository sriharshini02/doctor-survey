import streamlit as st
import pandas as pd
import joblib

# Load encoded data
data = pd.read_csv("processed_npi_data.csv")

# Load label encoders
label_encoders = joblib.load("label_encoders.pkl")

# Reverse the encoding mappings (convert numbers back to labels)
decoded_mappings = {
    col: {v: k for k, v in label_encoders[col].items()} for col in label_encoders
}

st.title("🩺 Doctor Survey Prediction App")

# User input section
st.markdown("### 🕒 Select the Time")
col1, col2 = st.columns(2)

with col1:
    hour = st.selectbox("Hour:", list(range(1, 13)), index=6)
with col2:
    ampm = st.radio("AM/PM:", ["AM", "PM"], horizontal=True)

# Convert hour to 24-hour format
if ampm == "PM" and hour != 12:
    hour += 12
elif ampm == "AM" and hour == 12:
    hour = 0

st.markdown("---")  # Adds a separator for better visual structure

# Filter doctors based on predicted time
filtered_data = data[data["Login Hour"] == hour].copy()

# Decode categorical values safely
for col in ["State", "Region", "Speciality"]:
    if col in filtered_data.columns:
        filtered_data.loc[:, col] = (
            filtered_data[col].map(decoded_mappings[col]).fillna("Unknown")
        )

if not filtered_data.empty:
    st.subheader("👨‍⚕️ Doctors Likely to Attend at This Time:")
    st.dataframe(
        filtered_data[["NPI", "State", "Region", "Speciality"]].reset_index(drop=True),
        use_container_width=True,
    )

    # Prepare CSV
    csv = filtered_data.to_csv(index=False).encode("utf-8")

    # Download Button
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="doctors_at_selected_time.csv",
        mime="text/csv",
    )
else:
    st.warning("⚠️ No doctors found for the selected time.")
