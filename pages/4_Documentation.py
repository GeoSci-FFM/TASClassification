import streamlit as st

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

st.header(":blue[Using TAS Classifier]")

st.subheader("Introduction")
st.markdown("""
The TAS Classifier is an ML-based tool designed for automatic rock classification using 10 major and minor oxides. By uploading your dataset, the app predicts rock types and visualizes them on a TAS (Total Alkali-Silica) plot. The classifier supports 3 distinct cases, allowing users to explore scenarios where certain major oxides (e.g., SiO₂ or alkali oxides) are excluded.
""")

st.subheader("How to Use the Program")
st.markdown("""
1. **Click on the "Upload CSV" button** to upload your dataset.
2. Ensure the file contains oxide values (e.g., SiO₂, MgO, CaO, etc.). A sample template is available for reference.
3. Use the toggle to choose between:
   - **Case 1**: All oxides.
   - **Case 2**: Excluding SiO₂.
   - **Case 3**: Excluding alkali oxides (Na₂O + K₂O).
4. The app verifies that your dataset includes the required oxides for the selected case. Missing or incorrect columns will prompt an error message.
5. The data is rearranged automatically to match the model’s expected column order.
6. Once validated, the model predicts rock types and appends the results as a new column in your dataset.
7. A preview of the updated dataset is displayed. The app plots the predicted rock types on a TAS diagram for visual interpretation.
""")

# Section: Fake Data Creation
st.header(":blue[Creating Synthetic Data]")

st.subheader("Introduction")
st.markdown("""
The Synthetic Data Creation module enables users to create fake geochemical data based on real-world data. Using statistical techniques such as mean and covariance analysis, the app creates realistic geochemical data that mirrors the variability and correlations observed in real samples. Users can visualize correlation matrices for both real and generated data, as well as download the fake data.
""")

st.subheader("How to Use the Program")
st.markdown("""
1. **Upload your real-world geochemical data** in CSV format. Ensure it includes oxide values and a rock type column.
2. **Click the "Produce Synthetic Data" button** to create fake geochemical data.
3.The tool analyzes the input data and generates synthetic data by replicating the statistical patterns
4. The tool then calculates and displays correlation matrices for both real and synthetic data. It also computes the difference between these matrices to validate the quality of the generated synthetic data.
5. **Download the synthetic data** in CSV format using the "Download Synthetic Data as CSV" button.
""")

# Footer for code link
st.markdown("""
### :blue[The entire code is available at:]
[Add GitHub Link Here](#)
""")
# st.header(':blue[Using TAS Classifier]')
# st.subheader('Introduction')
# st.write('The TAS Classifier is a ML based tool designed for automatic rock classification using 10 major and minor oxides. By uploading your dataset, the app predicts rock types and visualises them on a TAS (Total Alkali-Silica) plot. The classifier supports 3 distinct cases, allowing users to explore scenarios where certain major oxides (e.g., SiO₂ or alkali oxides) are excluded.')
# st.subheader('How to Use the Program')
# st.write('1. Click on the "Upload CSV" button to upload your dataset.')
# st.write('2. Ensure the file contains oxide values, e.g., SiO₂, MgO, CaO, etc. A sample template is available for reference.')
# st.write('3. Use the toggle to choose between Case 1 (All oxides), Case 2 (Excluding SiO₂), or Case 3 (Excluding alkali oxides: Na₂O + K₂O).')
# st.write('4. The app verifies that your dataset includes the required oxides for the selected case. Missing or incorrect columns will prompt an error message.')
# st.write('5. The data is rearranged automatically to match the model’s expected column order.')
# st.write('6. Once validated, the model predicts rock types and appends the results as a new column in your dataset.')
# st.write('7. A preview of the updated dataset is displayed. The app plots the predicted rock types on a TAS diagram for visual interpretation.')
# st.header(':blue[Creating Fake Data]')
# st.subheader('Introduction')
# st.write('The Fake Data Creation module enables users to create fake geochemical data based on real-world data. Using statistical techniques, such as mean and covariance analysis, the app creates realistic geochemical data that mirrors the variability and correlations observed in real samples. Users can visualize correlation matrices for both real and generated data as well as download the fake data.')
# st.subheader('How to Use the Program')
# st.write('1. Upload your real-world geochemical data in CSV format. Ensure it includes oxide values and a rock type column.')
# st.write('2. Click the "Produce Fake Data" button to create fake geochemical data.')
# st.write('3. The tool uses statistical techniques (e.g., mean and covariance calculations) to create realistic data for each rock type, maintaining interdependencies and variability.')
# st.write('4. The app calculates and displays correlation matrices for both real and fake data. It also computes the difference between these matrices to validate the quality of the generated fake data.')
# st.write('5. You can then download the fake data in CSV format using the "Download Fake Data as CSV" button.')
# st.subheader(' :blue[The entire code is available at -add link- ]')
