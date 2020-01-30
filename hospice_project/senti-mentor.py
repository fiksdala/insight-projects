import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

#%%
# Read in Data
disp_df = pd.read_pickle('data/processed/disp_df.pickle')

# Dependent Variable
dv = 'RECOMMEND_BBV'

#%% Display Settings
# Side bar stuff
state = st.sidebar.selectbox(
    'State',
    sorted(disp_df['State'].unique())
)

facility = st.sidebar.selectbox(
    'Facility',
    sorted(list(disp_df[disp_df['State'] == state]['Facility Name']))
)

if disp_df.loc[
    disp_df['Facility Name'] == facility
    ].loc[disp_df['State'] == state].shape[0] > 1:
    ccn = st.sidebar.selectbox(
        f'Multiple records for {facility}. Please select CCN.',
        disp_df.loc[
            disp_df['Facility Name'] == facility
            ].loc[disp_df['State'] == state]['ccn'].to_numpy()
    )
else:
    ccn = disp_df.loc[
            disp_df['Facility Name'] == facility
            ].loc[disp_df['State'] == state]['ccn'].to_numpy()[0]

display_type = st.sidebar.selectbox(
    'Display Type',
    np.array(['State Distribution', 'National Distribution'])
)

# Title
st.title('Welcome to Senti-Mentor')
st.write('Helping hospices visualize patient satisfaction')

# Main plot mask
show_state_warning = False
if display_type == 'State Distribution':
    if sum(~disp_df[disp_df['State'] == state][dv].isna())==0:
        show_state_warning = True
        mask = ~disp_df[dv].isna()
    else:
        mask = (disp_df['State'] == state) & (~disp_df[dv].isna())

else:
    mask = ~disp_df[dv].isna()

# Main Plot
main_distplot = ff.create_distplot(
    [[i for i in disp_df[mask]['RECOMMEND_BBV']]],
    [display_type]
)
st.plotly_chart(main_distplot)
if show_state_warning:
    st.write('''No facilities in selected state have relevant scores, showing 
    national distribution.
    ''')

# Prediction
