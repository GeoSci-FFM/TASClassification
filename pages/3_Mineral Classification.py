#!/usr/bin/env python
# coding: utf-8

# In[ ]:
#importing the necessary libraries.
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from tensorflow.keras.models import load_model

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url('https://raw.githubusercontent.com/tamanna1312/TASClassification/main/Applogo.jpg');
                background-repeat: no-repeat;
                background-size: 150px 150px; /* Set explicit width and height */
                background-position: 30px 10px; /* Position it in the top left */
                # margin-top: 20px; /* Add space above */
                padding-top: 170px; /* Add space below to separate from text */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_logo()

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

st.sidebar.image(
    'Goethe-Logo.gif')


st.title("Mineral Classifier")
#title of the app.
st.subheader("What is Two-Step Mineral Classifier?")
st.markdown(
    """
    The **Two-Step Mineral Classifier** is a machine learning (ML) tool designed for automated mineral identification from geochemical composition data.
    
     - **Dataset**: Upload your oxide wt% dataset.
     - **Step 1 – Mineral Group Prediction**: An XGBoost model predicts the mineral group (e.g., silicates, oxides, sulfides, etc.).
     - **Step 2 – Mineral Name Prediction**: A group-specific neural network model predicts the final mineral species within the predicted group.

   
    # **Supported Cases**
    # 1. **All Oxides**: Uses all 10 major and minor oxides.
    # 2. **No SiO₂**: Excludes SiO₂.
    # 3. **No Alkali Oxides**: Excludes Na₂O and K₂O.

    # **Results** 
    # - The app validates your data by checking the element oxides requirements according to the case selected, predicts rock types, and then displays:
    #    1. A table with your dataset and the predicted rock types.
    #    2. A TAS plot showing the classification visually.
   
    # """
)


    # - **Dataset**: Upload your oxide wt% dataset.
    # - **Step 1 – Mineral Group Prediction**: An XGBoost model predicts the mineral group (e.g., silicates, oxides, sulfides, etc.).
    # - **Step 2 – Mineral Name Prediction**: A group-specific neural network model predicts the final mineral species within the predicted group.

    # **How It Works**
    # 1. The app cleans and standardises the uploaded data.
    # 2. Elemental features are aligned with the training dataset.
    # 3. The model first predicts the mineral group.
    # 4. A dedicated model for that group predicts the final mineral name.

    # **Results**
    # - The app returns:
    #    1. A table containing your dataset with:
    #       - Predicted mineral group
    #       - Predicted mineral name
    #    2. A downloadable CSV file with the full classification results.

# st.write("Upload a CSV file with elemental wt% data to predict mineral group and mineral name.")

# -----------------------------
# Data Cleaning Functions
# -----------------------------

# def clean_cell(val):
#     if isinstance(val, str):
#         val = val.strip()
#         if re.match(r'^<\s*\d*\.?\d+$', val):
#             return 0.0
#         if val.lower() in ['na', 'n/a', 'nan', 'none', '']:
#             return np.nan
#         if val.startswith('#'):
#             return np.nan
#         val = val.replace(',', '.')
#         if re.search(r'[\\/;]', val):
#             return np.nan
#     try:
#         return float(val)
#     except:
#         return val

# def clean_dataframe(df):
#     df = df.copy()
#     numeric_cols = df.select_dtypes(include=['object']).columns
#     df[numeric_cols] = df[numeric_cols].applymap(clean_cell)
#     df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='ignore')

#     drop_cols = [
#         'mineral_frequency', 'sample_label', 'rock_name', 'classification',
#         'latitude', 'longitude', 'doi/ref', 'igsn', 'analytical_method', 'data_source'
#     ]
#     df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
#     return df

# # -----------------------------
# # Load Step 1 Model (cached)
# # -----------------------------

# @st.cache_resource
# def load_step1():
#     xgb_model = joblib.load("mineral_group_xgb_model.pkl")
#     encoder = joblib.load("group_label_encoder.pkl")
#     feature_columns = joblib.load("feature_columns.pkl")
#     return xgb_model, encoder, feature_columns

# # -----------------------------
# # File Upload
# # -----------------------------

# uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# if uploaded_file is not None:

#     st.success("File uploaded successfully!")

#     df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
#     st.write("Preview of uploaded data:")
#     st.dataframe(df.head())

#     # -----------------------------
#     # Clean Data
#     # -----------------------------

#     with st.spinner("Cleaning data..."):
#         unknown_clean = clean_dataframe(df)

#     # -----------------------------
#     # Step 1: Group Prediction
#     # -----------------------------

#     with st.spinner("Loading Step 1 model..."):
#         xgb_model, encoder, feature_columns = load_step1()

#     X_unknown = unknown_clean.reindex(columns=feature_columns, fill_value=0)

#     with st.spinner("Predicting mineral groups..."):
#         group_preds_encoded = xgb_model.predict(X_unknown)
#         group_preds = encoder.inverse_transform(group_preds_encoded)

#     unknown_clean["predicted_group"] = group_preds

#     st.success("Step 1 complete — Mineral groups predicted!")
#     st.dataframe(unknown_clean[["predicted_group"]].head())

#     # -----------------------------
#     # Step 2: Mineral Prediction
#     # -----------------------------

#     final_predictions = []

#     st.write("### Step 2: Predicting minerals per group")

#     for group in unknown_clean["predicted_group"].unique():
#         try:
#             st.write(f"Processing group: {group}")

#             group_data = unknown_clean[
#                 unknown_clean["predicted_group"] == group
#             ].copy()

#             X_group = group_data[feature_columns].fillna(0)

#             # Load per-group model files
#             scaler_path = f"scaler_{group}.pkl"
#             model_path = f"model_{group}.h5"
#             class_names_path = f"class_names_{group}.pkl"

#             if not (os.path.exists(scaler_path) and 
#                     os.path.exists(model_path) and 
#                     os.path.exists(class_names_path)):
#                 st.warning(f"Missing files for group {group}")
#                 continue

#             scaler = joblib.load(scaler_path)
#             class_names = joblib.load(class_names_path)
#             model = load_model(model_path)

#             # Scale & Predict
#             X_scaled = scaler.transform(X_group)
#             pred_probs = model.predict(X_scaled)
#             pred_labels = np.argmax(pred_probs, axis=1)
#             mineral_preds = [class_names[i] for i in pred_labels]

#             group_data["predicted_mineral"] = mineral_preds
#             final_predictions.append(group_data)

#         except Exception as e:
#             st.error(f"Error predicting for {group}: {e}")

#     # -----------------------------
#     # Final Output
#     # -----------------------------

#     if final_predictions:
#         result_df = pd.concat(final_predictions)

#         st.success("Full pipeline complete!")
#         st.write("### Final Predictions")
#         st.dataframe(result_df.head())

#         # Download button
#         csv = result_df.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             label="Download Final Predictions CSV",
#             data=csv,
#             file_name="final_pipeline_predictions.csv",
#             mime="text/csv"
#         )

#     else:
#         st.error("No valid predictions — check model/scaler files or input data.")
