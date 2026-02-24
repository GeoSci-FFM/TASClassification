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
