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

   
      **How It Works**
       1. The app cleans and standardises the uploaded data.
       2. Elemental features are aligned with the training dataset.
       3. The model first predicts the mineral group.
       4. A dedicated model for that group predicts the final mineral name.

     **Results**
       - The app returns:
       1. A table containing your dataset with:
          - Predicted mineral group
          - Predicted mineral name
       2. A downloadable CSV file with the full classification results.

   
    # """
)
test_data_path = "TestDataMineralModel.csv"  
st.write('See template to upload your data.')
template_file_path = "Template.csv"
template_data = pd.read_csv(template_file_path)
with st.expander("View Template File"):
    st.dataframe(template_data)
st.write('You can use test data for demo or upload your own csv file.')
use_test_data = st.toggle("Test data")
# -----------------------------
# Data Cleaning Functions
# -----------------------------

def clean_cell(val):
    if isinstance(val, str):
        val = val.strip()
        if re.match(r'^<\s*\d*\.?\d+$', val):
            return 0.0
        if val.lower() in ['na', 'n/a', 'nan', 'none', '']:
            return np.nan
        if val.startswith('#'):
            return np.nan
        val = val.replace(',', '.')
        if re.search(r'[\\/;]', val):
            return np.nan
    try:
        return float(val)
    except:
        return val

def clean_dataframe(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['object']).columns
    df[numeric_cols] = df[numeric_cols].applymap(clean_cell)
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='ignore')

    drop_cols = [
        'mineral_frequency', 'sample_label', 'rock_name', 'classification',
        'latitude', 'longitude', 'doi/ref', 'igsn', 'analytical_method', 'data_source'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

# -----------------------------
# Load Step 1 Model (cached)
# -----------------------------

@st.cache_resource
def load_step1():
    xgb_model = joblib.load("mineral_group_xgb_model.pkl")
    encoder = joblib.load("group_label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return xgb_model, encoder, feature_columns

# -----------------------------
# File Upload
# -----------------------------

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    st.success("File uploaded successfully!")

    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    # -----------------------------
    # Enforce Column Order (ADD HERE)
    # -----------------------------

    columns_order = [
        "mineral_name","mineral_group","mineral_frequency","sample_label","rock_name","classification",
        "latitude","longitude","doi/ref","igsn","analytical_method","data_source",
        "SiO2","TiO2","Al2O3","FeO","MnO","MgO","Cr2O3","Fe2O3","CaO","Na2O","K2O","P2O5","NiO",
        "BaO","CO2","SO3","SO2","PbO","SrO","ZrO2","Nb2O5","B2O3","WO3","As2O5","ZnO","MoO3","CuO",
        "CdO","Mn2O3","Cu2O","SnO","BeO","SnO2","H2O","F","Cl","Si","Ti","Al","Fe","S","C","Cu","Pb",
        "Zn","Co","Ni","As","Ag","Sb","Hg","Bi","Te","Mo","Mn","Mg","Ca","Na","K","Cr","Sr","Ba",
        "Y2O3","Sc2O3","La2O3","Ce2O3","Pr2O3","Nd2O3","Sm2O3","Gd2O3","Dy2O3","ThO2","UO2","Tb2O3",
        "V2O5","Li","PbO2","TeO2","V2O3","MnO2","Li2O","Cs2O","GeO2","Rb2O","NH42O","Ti2O3"
    ]

    # Strict schema alignment (recommended for MineralTD work)
    df = df.reindex(columns=columns_order)

    st.write("Preview of uploaded data (schema-aligned):")
    st.dataframe(df.head())

    # -----------------------------
    # Clean Data
    # -----------------------------

    with st.spinner("Cleaning data..."):
        unknown_clean = clean_dataframe(df)
        
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
