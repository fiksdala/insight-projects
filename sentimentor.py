import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from PIL import Image
import pickle
import numpy as np

@st.cache(allow_output_mutation=True)
def load_data():
    complete_df = pd.read_pickle('models/complete_df.pickle')
    lr_est_rating = pd.read_pickle('models/lr_est_rating.pickle')
    lr_recommend = pd.read_pickle('models/lr_recommend.pickle')
    pipe_est_rating = pd.read_pickle('models/pipe_est_rating.pickle')
    pipe_recommend = pd.read_pickle('models/pipe_recommend.pickle')
    sparse_preds = pd.read_pickle('models/sparse_preds.pickle')
    return complete_df, lr_est_rating, lr_recommend, pipe_est_rating, \
        pipe_recommend, sparse_preds


complete_df, lr_est_rating, lr_recommend, pipe_est_rating, \
pipe_recommend, sparse_preds = load_data()

coef_labels = ['Number of Providers within 60 Miles',
 'Provider Visit Within 3 Days of Death',
 'Team Met Emotional Needs',
 'Team Managed Symptoms',
 'Team Communicated Effectively',
 'Care Delivered On Time',
 'Staff Adequately Trained',
 '% Patients < 30 days',
 'Nurse Visit per Patient',
 'Social Work Visit per Patient',
 'Physician Visit per Patient',
 '% Rural Zip',
 'For-Profit']

insight_vars = ['Provider Visit Within 3 Days of Death',
 'Team Met Emotional Needs',
 'Team Managed Symptoms',
 'Team Communicated Effectively',
 'Care Delivered On Time',
 'Staff Adequately Trained',
 '% Patients < 30 days',
 'Nurse Visit per Patient',
 'Social Work Visit per Patient',
 'Physician Visit per Patient']

image_wr = Image.open('models/would_recommend_shap.png')
image_er = Image.open('models/rate_xgb.png')

sparse_rec_mae = 1.8
sparse_rate_mae = .2

#%% Display Settings
# Side bar stuff
state = st.sidebar.selectbox(
    'State',
    sorted(complete_df['State'].unique())
)

facility = st.sidebar.selectbox(
    'Facility',
    sorted(list(complete_df[complete_df['State'] == state]['Facility Name']))
)

if complete_df.loc[
    complete_df['Facility Name'] == facility
    ].loc[complete_df['State'] == state].shape[0] > 1:
    ccn = st.sidebar.selectbox(
        f'Multiple records for {facility}. Please select CCN.',
        complete_df.loc[
            complete_df['Facility Name'] == facility
            ].loc[complete_df['State'] == state]['ccn'].to_numpy()
    )
else:
    ccn = complete_df.loc[
            complete_df['Facility Name'] == facility
            ].loc[complete_df['State'] == state]['ccn'].to_numpy()[0]


if ccn in [str(i) for i in sparse_preds['ccn']]:
    sparse = True
    ccn_recommend = sparse_preds[sparse_preds['ccn'] == ccn]['would_recommend']
    ccn_est_rating = sparse_preds[sparse_preds['ccn'] == ccn]['RATING_EST']
    ccn_recommend = ccn_recommend.to_numpy()[0]
    ccn_est_rating = ccn_est_rating.to_numpy()[0]
    # st.write('SPARSE')
else:
    sparse = False
    ccn_recommend = complete_df.loc[complete_df['ccn'] == ccn,
                                    'would_recommend']
    ccn_recommend = ccn_recommend.to_numpy()[0]
    ccn_est_rating = complete_df.loc[complete_df['ccn'] == ccn,
                                    'RATING_EST']
    ccn_est_rating = ccn_est_rating.to_numpy()[0]
    # st.write('FULL')

# Title
st.title('Welcome to Senti-Mentor')
st.write('*Helping hospices improve patient and family satisfaction*')

# Specify view type
main_view_type = st.radio(
    'Select View Type',
    ['Comparisons',
     'Model Summary',
     'Targeted Recommendations']
)

if main_view_type=='Comparisons':
    st.write("**Compare your facility's performance with at the state or national"
             " level below (your score is the red line).**")
    comparison_view = st.selectbox(
        'Select Comparison Type',
        ['National', 'State']
    )
    measure = st.radio(
        '',
        ['Recommendation Percentage', 'Rating']
    )
    if measure=='Recommendation Percentage':
        if sparse:
            st.write('**This facility is missing key features, including the'
                     '  recommendation percentage. Your indicated score '
                     ' is based on the *Senti-Mentor '
                     ' Prediction Model*, which has a mean absolute error of '
                     f' {sparse_rec_mae}%.**')
        dv = 'would_recommend'
        bin_size = 1
        dv_val = ccn_recommend
        y1 = .2
    else:
        if sparse:
            st.write('**This facility is missing key features, including the'
                     '  overall rating estimate. Your indicated score '
                     ' is based on the *Senti-Mentor '
                     ' Prediction Model*, which has a mean absolute error of '
                     f' {sparse_rate_mae} points.**')
        dv = 'RATING_EST'
        bin_size = .1
        dv_val = ccn_est_rating
        y1 = 2
    mask = ~complete_df[dv].isna()
    if comparison_view=='National':
        main_distplot = ff.create_distplot(
            [[i for i in complete_df[mask][dv]]],
            [comparison_view],
            bin_size=bin_size
        )
        main_distplot.add_shape(
            # Line Vertical
            go.layout.Shape(
                type="line",
                x0=dv_val,
                y0=0,
                x1=dv_val,
                y1=y1,
                line=dict(
                    color="Red",
                    width=3
                )
            ))
        main_distplot.update_layout(
            title_text=f'Distribution of {measure}',
            xaxis_title=measure,
            yaxis_title='Density'
        )
        st.plotly_chart(main_distplot)

    if comparison_view=='State':
        mask = (complete_df['State']==state) & (~complete_df[dv].isna())
        if sum(mask)==0:
            st.write(f'**Not enough values in {state} to create a state '
                     f' distribution, choose *National* comparison type'
                     f'  instead.**')
        else:
            main_distplot = ff.create_distplot(
                [[i for i in
                  complete_df[mask][dv]]],
                [comparison_view],
                bin_size=bin_size
            )
            main_distplot.add_shape(
                # Line Vertical
                go.layout.Shape(
                    type="line",
                    x0=dv_val,
                    y0=0,
                    x1=dv_val,
                    y1=y1,
                    line=dict(
                        color="Red",
                        width=3
                    )
                ))
            main_distplot.update_layout(
                title_text=f'Distribution of {measure}',
                xaxis_title=measure,
                yaxis_title='Density'
            )
            st.plotly_chart(main_distplot)


if main_view_type == 'Model Summary':
    model_view = st.selectbox(
        'Model',
        ['Recommendations', 'Ratings']
    )
    if model_view == 'Recommendations':
        disp_model = lr_recommend
        if sparse:
            st.subheader("Senti-Mentor Prediction Model")
            st.write("Model overview here")
            st.image(image_wr, width=600)
        else:
            st.subheader("Senti-Mentor Insight Model")
            st.write("Model overview here")
    else:
        disp_model = lr_est_rating
        if sparse:
            st.subheader("Senti-Mentor Prediction Model")
            st.write("Model overview here")
            st.image(image_er, width=600)
        else:
            st.subheader("Senti-Mentor Insight Model")
            st.write("Model overview here")

    if not sparse:
        fig = go.Figure(go.Bar(
            x=disp_model.coef_,
            # y=pipe_recommend['scaler'].colnames_,
            y=coef_labels,
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

raw_obs = complete_df.loc[complete_df['ccn']==ccn,
                          pipe_recommend['scaler'].colnames_]

show_columns = ['Your Score', 'Mean Score', 'Prediction Contribution']
def get_recs():
    trans_obs = pipe_recommend.transform(raw_obs)
    trans_obs_t = pd.DataFrame(trans_obs.to_numpy().transpose())
    pred_score = lr_recommend.predict(trans_obs)
    st.subheader('Performance Overview')
    st.write(f"Our model indicates that {round(pred_score[0],2)}% of families would"
             f" recommend a hospice with your facility's scores, on average."
             f" Your facility's actual score was {round(ccn_recommend,2)}%")
    st.write('The figure below indicates how factors influenced your *predicted*'
             ' score. Keep in'
             ' mind that does not necessarily mean those factors *directly* caused'
             ' your score to increase or decrease (regression models cannot '
             ' establish causation on their own!). '
             ' Specific recommendations '
             ' and additional context are listed below the figure.')
    coef_df = pd.DataFrame({
        'raw_feature': pipe_recommend['scaler'].colnames_,
        'Feature': coef_labels,
                            'Coefficient': lr_recommend.coef_})
    coef_df = pd.concat([coef_df, trans_obs_t], axis=1)
    coef_df = coef_df.set_index('Feature')
    your_score = pipe_recommend['scaler'].inverse_transform(trans_obs).transpose()
    coef_df['Your Score'] = your_score
    ave_scores = [complete_df[i].mean() for i in coef_df['raw_feature']]
    coef_df['Mean Score'] = ave_scores
    coef_df['Prediction Contribution'] = coef_df['Coefficient'] * coef_df[0]
    sort_table = coef_df.loc[
        insight_vars, show_columns].sort_values('Prediction Contribution', ascending=False)
    # st.table(sort_table)
    # st.write(sort_table['Prediction Contribution'])
    fig = go.Figure(go.Bar(
        x=sort_table['Prediction Contribution'],
        # y=pipe_recommend['scaler'].colnames_,
        y=sort_table.index,
        orientation='h'
    ))
    fig.update_layout(
        title='Contribution to Predicted Score',
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

    def lh(val):
        if val >0:
            return 'raised'
        else:
            return 'lowered'

    st.write(f"Your facility benefited most from the *{sort_table.index[0]}* "
             f" factor, which"
             f" **{lh(sort_table['Prediction Contribution'][0])} "
             f" your predicted score "
             f" by {round(sort_table['Prediction Contribution'][0], 2)}%"
             f" compared to the average.**")
    st.write(f"The factor that benefited your facility the least was"
             f" *{sort_table.index[-1]}*, which **resulted in a difference of"
             f" {round(sort_table['Prediction Contribution'][-1], 2)}"
             f" compared to the average.**")

    st.write(f"Combined, your 3 lowest performing factors ("
             f"*{sort_table.index[-3]}*, *{sort_table.index[-2]}*, "
             f" and *{sort_table.index[-1]}*) "
             f"**{lh(sum(sort_table['Prediction Contribution'][-3:]))}"
             f" your predicted score by "
             f" {round(sum(sort_table['Prediction Contribution'][-3:]),2)}"
             f" % compared to the average.**")

    st.subheader('Recommendations: How to use this information?')
    st.write('This analysis relied on regression techniques that **cannot'
             ' establish causal relationships.** Additionally, some '
             ' features described here may not be good targets for direct'
             ' intervention, either because they are not directly modifiable'
             ' or it would be counterproductive to do so. '
             ' **Still, these insights'
             ' can be used to help identify and prioritize potential '
             ' targets for intervention for improvements.** '
             ' The 3 lowest performing factors listed above represent the'
             ' best "bang for your buck" in terms of potential improvement,'
             ' since they balance the impact of the predictor with '
             ' your specific score and room for improvement.'
             ' A sound approach would be to interrogate your processes '
             'related to those factors outlined above in order to '
             ' identify potential root causes of success or failure. For an '
             ' even more data-driven'
             ' approach, a properly-designed experiment may be able to '
             ' shed further light on how your processes impact performance.'
             )


def get_rates():
    trans_obs = pipe_est_rating.transform(raw_obs)
    trans_obs_t = pd.DataFrame(trans_obs.to_numpy().transpose())
    pred_score = lr_est_rating.predict(trans_obs)
    st.subheader('Performance Overview')
    st.write(f"Our model indicates that families would rate your hospice"
             f" {round(pred_score[0],2)} out of 10, on average. "
             f" Your facility's actual score was {round(ccn_est_rating,2)}")
    st.write('The figure below indicates how factors influenced your *predicted*'
             ' score. Keep in'
             ' mind that does not necessarily mean those factors *directly* caused'
             ' your score to increase or decrease (regression models cannot '
             ' establish causation on their own!). '
             ' Specific recommendations '
             ' and additional context are listed below the figure.')
    coef_df = pd.DataFrame({
        'raw_feature': pipe_est_rating['scaler'].colnames_,
        'Feature': coef_labels,
                            'Coefficient': lr_est_rating.coef_})
    coef_df = pd.concat([coef_df, trans_obs_t], axis=1)
    coef_df = coef_df.set_index('Feature')
    your_score = pipe_est_rating['scaler'].inverse_transform(
        trans_obs).transpose()
    coef_df['Your Score'] = your_score
    ave_scores = [complete_df[i].mean() for i in coef_df['raw_feature']]
    coef_df['Mean Score'] = ave_scores
    coef_df['Prediction Contribution'] = coef_df['Coefficient'] * coef_df[0]
    sort_table = coef_df.loc[
        insight_vars, show_columns].sort_values('Prediction Contribution',
                                                ascending=False)
    # st.table(sort_table)
    # st.write(sort_table['Prediction Contribution'])
    fig = go.Figure(go.Bar(
        x=sort_table['Prediction Contribution'],
        # y=pipe_recommend['scaler'].colnames_,
        y=sort_table.index,
        orientation='h'
    ))
    fig.update_layout(
        title='Contribution to Predicted Score',
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

    def lh(val):
        if val >0:
            return 'raised'
        else:
            return 'lowered'

    st.write(f"Your facility benefited most from the *{sort_table.index[0]}* "
             f" factor, which"
             f" **{lh(sort_table['Prediction Contribution'][0])} "
             f" your predicted score "
             f" by {round(sort_table['Prediction Contribution'][0], 2)} points"
             f" compared to the average.**")
    st.write(f"The factor that benefited your facility the least was"
             f" *{sort_table.index[-1]}*, which **resulted in a difference of"
             f" {round(sort_table['Prediction Contribution'][-1], 2)}"
             f" points compared to the average.**")

    st.write(f"Combined, your 3 lowest performing factors ("
             f"*{sort_table.index[-3]}*, *{sort_table.index[-2]}*, "
             f" and *{sort_table.index[-1]}*) "
             f"**{lh(sum(sort_table['Prediction Contribution'][-3:]))}"
             f" your predicted score by "
             f" {round(sum(sort_table['Prediction Contribution'][-3:]),2)}"
             f" points out of ten compared to the average.**")

    st.subheader('Recommendations: How to use this information?')
    st.write('This analysis relied on regression techniques that **cannot'
             ' establish causal relationships.** Additionally, some '
             ' features described here may not be good targets for direct'
             ' intervention, either because they are not directly modifiable'
             ' or it would be counterproductive to do so. '
             ' **Still, these insights'
             ' can be used to help identify and prioritize potential '
             ' targets for intervention for improvements.** '
             ' The 3 lowest performing factors listed above represent the'
             ' best "bang for your buck" in terms of potential improvement,'
             ' since they balance the impact of the predictor with '
             ' your specific score and room for improvement.'
             ' A sound approach would be to interrogate your processes '
             'related to those factors outlined above in order to '
             ' identify potential root causes of success or failure. For an '
             ' even more data-driven'
             ' approach, a properly-designed experiment may be able to '
             ' shed further light on how your processes impact performance.'
             )


if main_view_type == 'Targeted Recommendations':
    model_view = st.selectbox(
            'Model',
            ['Recommendations', 'Ratings']
        )
    if model_view == 'Recommendations':
        if sparse:
            st.write(f"Your facility's predicted family recommendation rate is "
                     f"{round(ccn_recommend)}%; "
                     "however, key modifiable features are missing. Targeted "
                     "recommendations are not available. For more information "
                     "on what factors informed your predicted family "
                     "recommendation percentage, see 'Model Summary'.")
        else:
            get_recs()
    if model_view == 'Ratings':
        if sparse:
            st.write(f"Your facility's predicted rating is "
                     f"{round(ccn_est_rating)}; "
                     "however, key modifiable features are missing. Targeted "
                     "recommendations are not available. For more information "
                     "on what factors informed your predicted "
                     "rating, see 'Model Summary'.")
        else:
            get_rates()
