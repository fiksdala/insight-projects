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
# Read in full, sparse, and ccn data

# @st.cache
def get_data():
    X_full = pd.read_pickle('data/interim/X_full.pickle')
    X_sparse = pd.read_pickle('data/interim/X_sparse.pickle')
    X_id = pd.read_pickle('data/interim/X_id.pickle')
    full_pipe = pd.read_pickle('data/interim/pipe.pickle')
    sparse_pipe = pd.read_pickle('data/interim/pipe_sparse.pickle')
    full_knn = pd.read_pickle('data/interim/knn_ids.pickle')
    sparse_knn = pd.read_pickle('data/interim/knn_sparse.pickle')
    return X_full, X_sparse, X_id, full_pipe, sparse_pipe, full_knn, sparse_knn

X_full, X_sparse, X_id, full_pipe, sparse_pipe, full_knn, sparse_knn = get_data()

# Read in ols_object for predict
ols_object = pickle.load(open('data/interim/ols_obj.pickle', 'rb'))

# Dependent Variable
dv = 'RECOMMEND_BBV'

# View indicator
new_pred_view = False

#%% Display Settings
# Side bar stuff
state = st.sidebar.selectbox(
    'State',
    sorted(X_id['State'].unique())
)

facility = st.sidebar.selectbox(
    'Facility',
    sorted(list(X_id[X_id['State'] == state]['Facility Name']))
)

if X_id.loc[
    X_id['Facility Name'] == facility
    ].loc[X_id['State'] == state].shape[0] > 1:
    ccn = st.sidebar.selectbox(
        f'Multiple records for {facility}. Please select CCN.',
        X_id.loc[
            X_id['Facility Name'] == facility
            ].loc[X_id['State'] == state]['ccn'].to_numpy()
    )
else:
    ccn = X_id.loc[
            X_id['Facility Name'] == facility
            ].loc[X_id['State'] == state]['ccn'].to_numpy()[0]

# Model type
model_full = X_id[X_id['ccn'] == ccn]['model_group'].to_numpy() == 'full'

# ccn observation
if model_full:
    ccn_obs = X_full[X_full['ccn'] == ccn].drop(columns='ccn')
else:
    ccn_obs = X_sparse[X_sparse['ccn'] == ccn].drop(columns='ccn')

# ccn y
ccn_y = X_id[X_id['ccn'] == ccn][dv].to_numpy()[0]

# get raw obs and model predictions
if model_full:
    ccn_obs = pd.DataFrame(ccn_obs,
                           columns=full_pipe['scaler'].colnames_)
    ccn_obs_raw_scale = full_pipe['scaler'].inverse_transform(ccn_obs)
    ccn_obs_raw_scale = pd.DataFrame(ccn_obs_raw_scale,
                                     columns=full_pipe['scaler'].colnames_)
    ccn_pred = ols_object.get_prediction(ccn_obs).summary_frame()
else:
    ccn_obs = pd.DataFrame(ccn_obs,
                           columns=sparse_pipe['scaler'].colnames_)
    ccn_obs_raw_scale = sparse_pipe['scaler'].inverse_transform(ccn_obs)
    ccn_obs_raw_scale = pd.DataFrame(ccn_obs_raw_scale,
                                     columns=sparse_pipe['scaler'].colnames_)

# Title
st.title('Welcome to Senti-Mentor')
st.write('Helping hospices visualize patient satisfaction')

# Main plot mask
# Main Plot (Initial Display)
show_state_warning = False

# Specify view type
main_view_type = st.radio(
    'Select View Type',
    ['State', 'National', 'Model Summary']
)

if main_view_type == 'State':
    if sum(~X_id[X_id['State'] == state][dv].isna()) == 0:
        show_state_warning = True
        mask = ~X_id[dv].isna()
    else:
        mask = (X_id['State'] == state) & (~X_id[dv].isna())

else:
    mask = ~X_id[dv].isna()

if ((main_view_type == 'State') |
    (main_view_type == 'National')):
    main_distplot = ff.create_distplot(
        [[i for i in X_id[mask]['RECOMMEND_BBV']]],
        [main_view_type]
    )
    st.plotly_chart(main_distplot)
    if show_state_warning:
        st.write('''No facilities in selected state have relevant scores, showing
        national distribution.
        ''')

if main_view_type == 'Model Summary':
    st.write('Put Model Summary Stuff Here!!')
    st.header('How does your facility compare?')
    if model_full:
        st.subheader('This facility has patient survey data available.')
    else:
        st.subheader('This facility is missing key survey data. Predictions will contain more uncertainty.')
    st.text('''The figure above summarizes the impact of key factors that
predict performance. You can visualize how differences in these factors changes
performance on average by selecting features from the dropdown box below to
adjust. For context, the three facilities that most closely match your
selections are displayed. You can view specific attributes of these facilities
at the bottom of this page.''')

# Custom specifications
alter_features = st.multiselect(
    'Select Predictors to Alter',
    ccn_obs.columns
)
st.write(ccn_obs.columns)
# Make new df from custom specifications
feature_dict = {}
new_pred_df_raw = ccn_obs_raw_scale.copy()
if model_full:
    for f in alter_features:
        feature_dict[f] = st.slider(f,
                                    X_full[f].min(),
                                    X_full[f].max(),
                                    ccn_obs_raw_scale[f].to_numpy()[0])
        new_pred_df_raw[f] = feature_dict[f]

    new_pred_df_scaled = full_pipe.transform(new_pred_df_raw)
    new_pred = ols_object.get_prediction(new_pred_df_scaled).summary_frame()
# else:
#     for f in alter_features:
#         feature_dict[f] = st.slider(f,
#                                     X_sparse[f].min(),
#                                     X_sparse[f].max(),
#                                     ccn_obs_raw_scale[f].to_numpy()[0])
#         new_pred_df_raw[f] = feature_dict[f]
#
#     new_pred_df_scaled = sparse_pipe.transform(new_pred_df_raw)
#     new_pred = ols_object.get_prediction(new_pred_df_scaled).summary_frame()

# # transform based on sparse/full
# if model_full:
#     distances, indices = full_knn.kneighbors(new_pred_df_scaled)
#     knn_df = X_full.iloc[indices[0],:]
#     knn_preds = ols_object.get_prediction(
#         knn_df.drop(columns='ccn')).summary_frame()
# else:
#     distances, indices = sparse_knn.kneighbors(new_pred_df_scaled)
#     knn_df = X_full.iloc[indices[0], :]
#
# # 95% CIs for KNNs
# knn_cis = knn_preds['mean_ci_upper']-knn_preds['mean']
#
# # knn observed dv
# knn1obs = X_id[
#     X_id['ccn'] == knn_df['ccn'].to_numpy()[0]
#     ]['RECOMMEND_BBV'].to_numpy()[0]
# knn2obs = X_id[
#     X_id['ccn'] == knn_df['ccn'].to_numpy()[1]
#     ]['RECOMMEND_BBV'].to_numpy()[0]
# knn3obs = X_id[
#     X_id['ccn'] == knn_df['ccn'].to_numpy()[2]
#     ]['RECOMMEND_BBV'].to_numpy()[0]
#
# knn1name = X_id[
#     X_id['ccn']==knn_df['ccn'].to_numpy()[0]]['Facility Name'].to_numpy()[0]
# knn2name = X_id[
#     X_id['ccn']==knn_df['ccn'].to_numpy()[1]]['Facility Name'].to_numpy()[0]
# knn3name = X_id[
#     X_id['ccn']==knn_df['ccn'].to_numpy()[2]]['Facility Name'].to_numpy()[0]
#
# # Create Figure
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=[facility,
#                             knn1name,
#                             knn2name,
#                             knn3name],
#                          y=[new_pred['mean'].to_numpy()[0],
#                             knn_preds['mean'].to_numpy()[0],
#                             knn_preds['mean'].to_numpy()[1],
#                             knn_preds['mean'].to_numpy()[2]],
#                          error_y=dict(
#                              type='data',
#                              array=[
#                                  new_pred['mean_ci_upper']-new_pred['mean'],
#                                  knn_cis.to_numpy()[0],
#                                  knn_cis.to_numpy()[1],
#                                  knn_cis.to_numpy()[2]]),
#                          mode='markers',
#                          name='Predicted'))
#
# fig.add_trace(go.Scatter(x=[facility, knn1name, knn2name, knn3name],
#                          y=[ccn_y,
#                             knn1obs,
#                             knn2obs,
#                             knn3obs],
#                          mode='markers',
#                          name='Observed'))
# st.plotly_chart(fig)
# if ~model_full:
#     st.write('**Note:** This facility does not have an observed value for the specified performance metric.')
# #
# # # Comparison Table
# #
# #
