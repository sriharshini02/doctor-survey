import streamlit as st
import pandas as pd
import joblib

# Load the trained model and data
model = joblib.load("npi_rf_model.pkl")
data = pd.read_csv("processed_npi_data.csv")

st.title("Doctor Survey Prediction App")

# Get user input for time
login_hour = st.number_input(
    "Enter Login Hour (0-23):", min_value=0, max_value=23, step=1
)

if st.button("Get Likely Participants"):
    # Prepare input data
    input_data = data.copy()
    input_data["Login Hour"] = login_hour
    predictions = model.predict(
        input_data.drop(columns=["NPI", "Count of Survey Attempts"])
    )

    # Get NPIs of predicted positive cases
    selected_npis = data.loc[predictions == 1, "NPI"]

    # Create downloadable CSV
    csv = selected_npis.astype(str).to_csv(index=False)

    st.download_button(
        "Download CSV", data=csv, file_name="likely_doctors.csv", mime="text/csv"
    )
