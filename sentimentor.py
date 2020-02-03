import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pickle
import numpy as np


#%%
# Read in full, sparse, and ccn data


final_X_all = pd.read_pickle('final_X_all.pickle')
final_X_id = pd.read_pickle('final_X_id.pickle')
final_ols_vars = pd.read_pickle('final_ols_vars.pickle')
final_ols = pd.read_pickle('final_ols.pickle')
full_pipe = pd.read_pickle('final_pipe.pickle')
full_knn = pd.read_pickle('final_knn_ids.pickle')

################################################################################
# @st.cache
# def get_data():
#     X_full = pd.read_pickle('X_full.pickle')
#     X_id = pd.read_pickle('X_id.pickle')
#     full_pipe = pd.read_pickle('pipe.pickle')
#     full_knn = pd.read_pickle('knn_ids.pickle')
#     return X_full, X_id, full_pipe, full_knn
#
# X_full, X_id, full_pipe, full_knn = get_data()
#
# X_full_id = X_id[X_id['model_group']=='full']

X_full_raw = full_pipe['scaler'].inverse_transform(final_X_all[final_ols_vars])
X_full_raw = pd.DataFrame(X_full_raw,
                          columns=full_pipe['scaler'].colnames_)
# # Read in ols_object for predict
# ols_object = pickle.load(open('ols_obj.pickle', 'rb'))
#
# # Dependent Variable
dv = 'RECOMMEND_BBV'

var_names = ['Ratio of Bottom to Top Ratings, Emotional/Spiritual Support',
 'Ratio of Bottom to Top Ratings, Respect',
 'Ratio of Bottom to Top Ratings, Symptoms',
 'Ratio of Bottom to Top Ratings, Team Communication',
 'Ratio of Bottom to Top Ratings, Timely Care',
 'Ratio of Bottom to Top Ratings, Training',
'Hospice Visit When Death is Imminent',
 'Nurse Visit Ct. per Beneficiary',
 'Social Work Visit Ct. per Beneficiary',
 'Physician Visit Ct. per Beneficiary',
  '% Beneficiaries 30 or Fewer Days']


varname_dict = dict(zip(
    var_names,
    final_ols_vars
))

#%% Display Settings
# Side bar stuff
state = st.sidebar.selectbox(
    'State',
    sorted(final_X_id['State'].unique())
)

facility = st.sidebar.selectbox(
    'Facility',
    sorted(list(final_X_id[final_X_id['State'] == state]['Facility Name']))
)

if final_X_id.loc[
    final_X_id['Facility Name'] == facility
    ].loc[final_X_id['State'] == state].shape[0] > 1:
    ccn = st.sidebar.selectbox(
        f'Multiple records for {facility}. Please select CCN.',
        final_X_id.loc[
            final_X_id['Facility Name'] == facility
            ].loc[final_X_id['State'] == state]['ccn'].to_numpy()
    )
else:
    ccn = final_X_id.loc[
            final_X_id['Facility Name'] == facility
            ].loc[final_X_id['State'] == state]['ccn'].to_numpy()[0]


ccn_obs = full_pipe.transform(
    final_X_all[final_X_all['ccn'] == ccn][final_ols_vars]
)

ccn_y = final_X_id[final_X_id['ccn'] == ccn][dv].to_numpy()[0]

# ccn_obs = pd.DataFrame(ccn_obs,
#                            columns=full_pipe['scaler'].colnames_)
ccn_obs_raw_scale = full_pipe['scaler'].inverse_transform(ccn_obs)
ccn_obs_raw_scale = pd.DataFrame(ccn_obs_raw_scale,
                                 columns=full_pipe['scaler'].colnames_)
ccn_pred = final_ols.get_prediction(ccn_obs).summary_frame()

def recommender(obs):
    potential_improvement = []
    for var in obs:
        obs_pos = obs[var].to_numpy() > 0
        coef_pos = final_ols.params[var] > 0
        obs_neg = obs[var].to_numpy() < 0
        coef_neg = final_ols.params[var] < 0

        if obs_pos and coef_pos:
            potential_improvement.append(var)
        if obs_neg and coef_neg:
            potential_improvement.append(var)
    output = obs[potential_improvement].transpose()
    output.columns = ['obs']
    output = output.iloc[(-np.abs(output['obs'].values)).argsort()]
    # return sorted by absolute values
    return output.iloc[:3, :]

# Title
st.title('Welcome to Senti-Mentor')
st.write('Helping hospices visualize patient satisfaction')

# Specify view type
main_view_type = st.radio(
    'Select View Type',
    ['State', 'National',
     'Model Summary: Compare Similar Facilities and Visualize Change',
     'Top 3 Recommendations']
)

if main_view_type == 'State':
    if sum(~final_X_id[final_X_id['State'] == state][dv].isna()) == 0:
        show_state_warning = True
        mask = ~final_X_id[dv].isna()
    else:
        mask = (final_X_id['State'] == state) & (~final_X_id[dv].isna())

else:
    mask = ~final_X_id[dv].isna()

if ((main_view_type == 'State') |
    (main_view_type == 'National')):
    main_distplot = ff.create_distplot(
        [[i for i in final_X_id[mask]['RECOMMEND_BBV']]],
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
    main_distplot.update_layout(
        title_text='Distribution of Percentage of Negative Recommendations',
        xaxis_title='Percent Negative Recommendations (red line is facility value)',
        yaxis_title='Density'
    )
    st.plotly_chart(main_distplot)

if main_view_type == 'Model Summary: Compare Similar Facilities and Visualize Change':
    fig = go.Figure(go.Bar(
        x=final_ols.params,
        y=final_ols_vars,
        orientation='h'))
    fig.update_layout(
        title='Regression Model Coefficients (standardized)',
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
    st.write('''The figure above summarizes the impact of key service performance factors that predict performance. You can visualize how differences in these factors changes performance on average by selecting features from the dropdown box below to adjust. For context, the three facilities that most closely match your
selections are displayed.''')

    # Custom specifications
    alter_features = st.multiselect(
        'Select Predictors to Alter',
        var_names
    )

    feature_dict = {}
    new_pred_df_raw = ccn_obs_raw_scale.copy()
    for f in alter_features:
        feature_dict[varname_dict[f]] = st.slider(f,
                                    final_X_all[varname_dict[f]].min(),
                                    final_X_all[varname_dict[f]].max(),
                                    ccn_obs_raw_scale[varname_dict[f]].to_numpy()[0])
        new_pred_df_raw[varname_dict[f]] = feature_dict[varname_dict[f]]

    new_pred_df_scaled = full_pipe.transform(new_pred_df_raw)
    new_pred = final_ols.get_prediction(new_pred_df_scaled).summary_frame()

    distances, indices = full_knn.kneighbors(new_pred_df_scaled)
    knn_df = final_X_all.iloc[indices[0],:]
    knn_plus_obs = knn_df
    knn_df = knn_df[knn_df['ccn']!=ccn]

    knn_preds = final_ols.get_prediction(
        full_pipe.transform(knn_df[final_ols_vars])
    ).summary_frame()
    knn_cis = knn_preds['mean_ci_upper'] - knn_preds['mean']

    # knn observed dv
    knn1obs = final_X_id[
        final_X_id['ccn'] == knn_df['ccn'].to_numpy()[0]
        ]['RECOMMEND_BBV'].to_numpy()[0]
    knn2obs = final_X_id[
        final_X_id['ccn'] == knn_df['ccn'].to_numpy()[1]
        ]['RECOMMEND_BBV'].to_numpy()[0]
    knn3obs = final_X_id[
        final_X_id['ccn'] == knn_df['ccn'].to_numpy()[2]
        ]['RECOMMEND_BBV'].to_numpy()[0]

    knn1name = final_X_id[
        final_X_id['ccn']==knn_df['ccn'].to_numpy()[0]]['Facility Name'].to_numpy()[0]
    knn2name = final_X_id[
        final_X_id['ccn']==knn_df['ccn'].to_numpy()[1]]['Facility Name'].to_numpy()[0]
    knn3name = final_X_id[
        final_X_id['ccn']==knn_df['ccn'].to_numpy()[2]]['Facility Name'].to_numpy()[0]

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
    fig.update_layout(
        title='Model Predicted Negative Recommendation Percentages',
        yaxis_title='Percent Negative Recommendation',
        autosize=False,
        width=700,
        height=700
    )
    st.plotly_chart(fig)


if main_view_type == 'Top 3 Recommendations':
    st.header('How can your facility improve?')
    targets = recommender(ccn_obs)
    your_score = ccn_obs_raw_scale[targets.index]
    changes = ccn_obs.copy()
    changes.loc[:, targets.index] = 0
    changes_transform = full_pipe['scaler'].inverse_transform(changes)
    changes_transform = pd.DataFrame(changes_transform,
                                     columns=ccn_obs.columns)
    changes_transform = changes_transform[targets.index]
    target_changes = pd.concat([your_score.transpose(),
                                changes_transform.transpose().iloc[:,0]],
                               axis=1)
    target_changes.columns = ['Your Score', 'Your Recommended Score']
    target_changes['Change'] = target_changes['Your Recommended Score'] - \
                               target_changes['Your Score']

    rec_pred = final_ols.get_prediction(changes).summary_frame()
    rec_pred = rec_pred['mean'].to_numpy()[0]
    if rec_pred < 0:
        rec_pred = 0
    st.write(f"""Based on our model and your facility's scores, here are some potential
    targets for improvement. Based on avilable data, **{ccn_y}% of families are unsatistfied with your performance and would not recommend** your facility. Based on our model, **{round(rec_pred)}% of families would not recommend** a facility with *Your Recommended Score* on average.""")
    st.table(target_changes)
# st.subheader('Compare specific values of comparable facilities below.')
# knn_plus_obs['Facility Name'] = [facility, knn1name, knn2name, knn3name]
# knn_plus_obs['Would Not Recommend'] = [ccn_y, knn1obs, knn2obs, knn3obs]
# knn_plus_obs = knn_plus_obs[['Facility Name', 'Would Not Recommend']+[i for i in knn_plus_obs.columns if i not in ['Facility Name', 'Would Not Recommend']]]
#
# st.write(knn_plus_obs)