import streamlit as st
import pandas

st.title('Welcome to Senti-Mentor')

st.sidebar.selectbox(
    'State',
    np.array(['MN', 'MA', 'TX'])
)