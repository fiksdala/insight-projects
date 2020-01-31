import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pickle


#%%
# Read in full, sparse, and ccn data

# @st.cache
def get_data():
    X_full = pd.read_pickle('data/interim/X_full.pickle')
    X_id = pd.read_pickle('data/interim/X_id.pickle')
    full_pipe = pd.read_pickle('data/interim/pipe.pickle')
    full_knn = pd.read_pickle('data/interim/knn_ids.pickle')
    return X_full, X_id, full_pipe, full_knn

X_full, X_id, full_pipe, full_knn = get_data()

X_full_id = X_id[X_id['model_group']=='full']

X_full_raw = full_pipe['scaler'].inverse_transform(X_full.drop(columns='ccn'))
X_full_raw = pd.DataFrame(X_full_raw,
                          columns=full_pipe['scaler'].colnames_)
# Read in ols_object for predict
ols_object = pickle.load(open('data/interim/ols_obj.pickle', 'rb'))

# Dependent Variable
dv = 'RECOMMEND_BBV'

#%% Display Settings
# Side bar stuff
state = st.sidebar.selectbox(
    'State',
    sorted(X_full_id['State'].unique())
)

facility = st.sidebar.selectbox(
    'Facility',
    sorted(list(X_full_id[X_full_id['State'] == state]['Facility Name']))
)

if X_full_id.loc[
    X_full_id['Facility Name'] == facility
    ].loc[X_full_id['State'] == state].shape[0] > 1:
    ccn = st.sidebar.selectbox(
        f'Multiple records for {facility}. Please select CCN.',
        X_full_id.loc[
            X_full_id['Facility Name'] == facility
            ].loc[X_full_id['State'] == state]['ccn'].to_numpy()
    )
else:
    ccn = X_full_id.loc[
            X_full_id['Facility Name'] == facility
            ].loc[X_full_id['State'] == state]['ccn'].to_numpy()[0]


ccn_obs = X_full[X_full['ccn'] == ccn].drop(columns='ccn')
ccn_y = X_id[X_id['ccn'] == ccn][dv].to_numpy()[0]

ccn_obs = pd.DataFrame(ccn_obs,
                           columns=full_pipe['scaler'].colnames_)
ccn_obs_raw_scale = full_pipe['scaler'].inverse_transform(ccn_obs)
ccn_obs_raw_scale = pd.DataFrame(ccn_obs_raw_scale,
                                 columns=full_pipe['scaler'].colnames_)
ccn_pred = ols_object.get_prediction(ccn_obs).summary_frame()

# Title
st.title('Welcome to Senti-Mentor')
st.write('Helping hospices visualize patient satisfaction')

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
    main_distplot.add_shape(
        # Line Vertical
        go.layout.Shape(
            type="line",
            x0=ccn_y,
            y0=0,
            x1=ccn_y,
            y1=.2,
            line=dict(
                color="Red",
                width=3
            )
        ))
    st.plotly_chart(main_distplot)

if main_view_type == 'Model Summary':
    fig = go.Figure(go.Bar(
        x=ols_object.params,
        y=ols_object.params.index,
        orientation='h'))
    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
    )
    st.plotly_chart(fig)
    st.header('How does your facility compare?')
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

feature_dict = {}
new_pred_df_raw = ccn_obs_raw_scale.copy()
for f in alter_features:
    feature_dict[f] = st.slider(f,
                                X_full_raw[f].min(),
                                X_full_raw[f].max(),
                                ccn_obs_raw_scale[f].to_numpy()[0])
    new_pred_df_raw[f] = feature_dict[f]

new_pred_df_scaled = full_pipe.transform(new_pred_df_raw)
new_pred = ols_object.get_prediction(new_pred_df_scaled).summary_frame()

distances, indices = full_knn.kneighbors(new_pred_df_scaled)
knn_df = X_full.iloc[indices[0],:]
knn_plus_obs = knn_df
knn_df = knn_df[knn_df['ccn']!=ccn]
knn_preds = ols_object.get_prediction(
    knn_df.drop(columns='ccn')).summary_frame()
knn_cis = knn_preds['mean_ci_upper'] - knn_preds['mean']

# knn observed dv
knn1obs = X_id[
    X_id['ccn'] == knn_df['ccn'].to_numpy()[0]
    ]['RECOMMEND_BBV'].to_numpy()[0]
knn2obs = X_id[
    X_id['ccn'] == knn_df['ccn'].to_numpy()[1]
    ]['RECOMMEND_BBV'].to_numpy()[0]
knn3obs = X_id[
    X_id['ccn'] == knn_df['ccn'].to_numpy()[2]
    ]['RECOMMEND_BBV'].to_numpy()[0]

knn1name = X_id[
    X_id['ccn']==knn_df['ccn'].to_numpy()[0]]['Facility Name'].to_numpy()[0]
knn2name = X_id[
    X_id['ccn']==knn_df['ccn'].to_numpy()[1]]['Facility Name'].to_numpy()[0]
knn3name = X_id[
    X_id['ccn']==knn_df['ccn'].to_numpy()[2]]['Facility Name'].to_numpy()[0]

# Create Figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=[facility,
                            knn1name,
                            knn2name,
                            knn3name],
                         y=[new_pred['mean'].to_numpy()[0],
                            knn_preds['mean'].to_numpy()[0],
                            knn_preds['mean'].to_numpy()[1],
                            knn_preds['mean'].to_numpy()[2]],
                         error_y=dict(
                             type='data',
                             array=[
                                 new_pred['mean_ci_upper']-new_pred['mean'],
                                 knn_cis.to_numpy()[0],
                                 knn_cis.to_numpy()[1],
                                 knn_cis.to_numpy()[2]]),
                         mode='markers',
                         name='Predicted'))

fig.add_trace(go.Scatter(x=[facility, knn1name, knn2name, knn3name],
                         y=[ccn_y,
                            knn1obs,
                            knn2obs,
                            knn3obs],
                         mode='markers',
                         name='Observed'))
st.plotly_chart(fig)

st.subheader('Compare specific values of comparable facilities below.')
knn_plus_obs['Facility Name'] = [facility, knn1name, knn2name, knn3name]
knn_plus_obs['Would Not Recommend'] = [ccn_y, knn1obs, knn2obs, knn3obs]
st.write(knn_plus_obs[['Facility Name', 'Would Not Recommend']+[i for i in knn_plus_obs.columns if i not in ['Facility Name', 'Would Not Recommend']]])