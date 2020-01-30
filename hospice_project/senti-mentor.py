import streamlit as st
import pandas
import numpy as np

# Read in Data








st.title('Welcome to Senti-Mentor')

# Side bar stuff
state = st.sidebar.selectbox(
    'State',
    np.array(['MN', 'MA', 'TX'])
)

facility = st.sidebar.selectbox(
    'Facility',
    np.array(['Hospice 1', 'Hospice 2'])
)

display_type = st.sidebar.selectbox(
    'Display Type',
    np.array(['State Distribution', 'National Distribution'])
)

# Main Plot
