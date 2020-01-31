import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from hospice_project.data.transformer import MyScaler
from sklearn.impute import KNNImputer
import statsmodels.api as sm

#%%
st.header('Welcome to Senti-Mentor')
# Read in full and sparse data

# @st.cache
def get_data():
    disp_df = pd.read_pickle('data/interim/initial_df.pickle')
    return disp_df

disp_df = get_data()
st.write(disp_df)










#
#
#
# disp_df = get_data()
# ols_object = pickle.load(open('data/interim/ols_obj.pickle', 'rb'))
# scale_impute = pickle.load(open('data/interim/pipe.pickle', 'rb'))
# r_keep = pickle.load(open('data/interim/r_keep.pickle', 'rb'))
#
# # Neighbors
# knn_full = pickle.load(open('data/interim/knn_ids.pickle', 'rb'))
# knn_sparse = pickle.load(open('data/interim/knn_sparse.pickle', 'rb'))
# # pipe_sparse = pickle.load(open('data/interim/pipe_sparse.pickle', 'rb'))
# sparse_keep_vars = pd.read_pickle('data/interim/sparse_keep_vars.pickle')
#
# # Dependent Variable
# dv = 'RECOMMEND_BBV'
#
# # View indicator
# new_pred_view = False
#
# #%% Display Settings
# # Side bar stuff
# state = st.sidebar.selectbox(
#     'State',
#     sorted(disp_df['State'].unique())
# )
#
# facility = st.sidebar.selectbox(
#     'Facility',
#     sorted(list(disp_df[disp_df['State'] == state]['Facility Name']))
# )
#
# if disp_df.loc[
#     disp_df['Facility Name'] == facility
#     ].loc[disp_df['State'] == state].shape[0] > 1:
#     ccn = st.sidebar.selectbox(
#         f'Multiple records for {facility}. Please select CCN.',
#         disp_df.loc[
#             disp_df['Facility Name'] == facility
#             ].loc[disp_df['State'] == state]['ccn'].to_numpy()
#     )
# else:
#     ccn = disp_df.loc[
#             disp_df['Facility Name'] == facility
#             ].loc[disp_df['State'] == state]['ccn'].to_numpy()[0]
#
# # ccn details
# ccn_obs = scale_impute.transform(disp_df[disp_df['ccn'] == ccn][r_keep])
# ccn_y = disp_df[disp_df['ccn'] == ccn][dv].to_numpy()[0]
# st.write(ccn_y)
# ccn_obs = pd.DataFrame(ccn_obs,
#                        columns=scale_impute['scaler'].colnames_)
# ccn_obs_raw_scale = scale_impute['scaler'].inverse_transform(ccn_obs)
# ccn_obs_raw_scale = pd.DataFrame(ccn_obs_raw_scale,
#                        columns=scale_impute['scaler'].colnames_)
# ccn_pred = ols_object.get_prediction(ccn_obs).summary_frame()
#
# if np.isnan(ccn_y):
#     model_type = 'sparse'
# else:
#     model_type = 'full'
#
# #
#
# # display_type = st.sidebar.selectbox(
# #     'Display Type',
# #     np.array(['State Distribution', 'National Distribution',
# #               'Comparison View', 'Model Summary'])
# # )
#
# # Title
# st.title('Welcome to Senti-Mentor')
# st.write('Helping hospices visualize patient satisfaction')
#
# # Main plot mask
# # Main Plot (Initial Display)
# show_state_warning = False
#
# main_view_type = st.radio(
#     'Select View Type',
#     ['State', 'National', 'Model Summary']
# )
#
# if main_view_type == 'State':
#     if sum(~disp_df[disp_df['State'] == state][dv].isna()) == 0:
#         show_state_warning = True
#         mask = ~disp_df[dv].isna()
#     else:
#         mask = (disp_df['State'] == state) & (~disp_df[dv].isna())
#
# else:
#     mask = ~disp_df[dv].isna()
#
# if ((main_view_type == 'State') |
#     (main_view_type == 'National')):
#     main_distplot = ff.create_distplot(
#         [[i for i in disp_df[mask]['RECOMMEND_BBV']]],
#         [main_view_type]
#     )
#     st.plotly_chart(main_distplot)
#     if show_state_warning:
#         st.write('''No facilities in selected state have relevant scores, showing
#         national distribution.
#         ''')
#
# if main_view_type == 'Model Summary':
#     st.write('Put Model Summary Stuff Here!!')
#     st.header('How does your facility compare?')
#     if model_type == 'full':
#         st.subheader('This facility has patient survey data available.')
#     else:
#         st.subheader('This facility is missing key survey data. Predictions will contain more uncertainty.')
#     st.text('''The figure above summarizes the impact of key factors that
# predict performance. You can visualize how differences in these factors changes
# performance on average by selecting features from the dropdown box below to
# adjust. For context, the three facilities that most closely match your
# selections are displayed. You can view specific attributes of these facilities
# at the bottom of this page.''')
#
# # Display alternative
#
# alter_features = st.multiselect(
#     'Select Predictors to Alter',
#     ccn_obs.columns
# )
#
# feature_dict = {}
# new_pred_df_raw = ccn_obs_raw_scale.copy()
# for f in alter_features:
#     feature_dict[f] = st.slider(f,
#                                 disp_df[f].min(),
#                                 disp_df[f].max(),
#                                 ccn_obs_raw_scale[f].to_numpy()[0])
#     new_pred_df_raw[f] = feature_dict[f]
#
# new_pred_df_scaled = scale_impute.transform(new_pred_df_raw)
# new_pred = ols_object.get_prediction(new_pred_df_scaled).summary_frame()
#
# if st.button('Reset Prediction to Observed Values'):
#     new_pred = ccn_pred
#
# # transform based on sparse/full
# if model_type == 'sparse':
#     nn1 = pipe_sparse.transform(disp_df[sparse_keep_vars])
#
#
#
# # Create Figure
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=[facility, 'b', 'c', 'd'],
#                          y=[new_pred['mean'].to_numpy()[0],2,3,4],
#                          error_y=dict(
#                              type='data',
#                              array=[
#                                  new_pred['mean_ci_upper']-new_pred['mean'],
#                                  0.5,
#                                  1.5,
#                                  2]),
#                          mode='markers',
#                          name='Predicted'))
# fig.add_trace(go.Scatter(x=[facility, 'b', 'c', 'd'],
#                          y=[ccn_y,
#                             5,
#                             6,
#                             7],
#                          mode='markers',
#                          name='Observed'))
# st.plotly_chart(fig)
# if model_type == 'sparse':
#     st.write('**Note:** This facility does not have an observed value for the specified performance metric.')
#
# # Comparison Table
#
#
