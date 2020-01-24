import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#%%

# Read in Full Data
df = pickle.load(
    open('/Users/alex/Documents/Insight/insight-projects/data/interim/prelim_df.pickle',
         'rb'))
df['neg_rec'] = pickle.load(
    open('/Users/alex/Documents/Insight/insight-projects/data/interim/Y.pickle', 'rb'))
y_train = pickle.load(open('/Users/alex/Documents/Insight/insight-projects/data/interim/y_train.pickle', 'rb'))

# Get data
X_train = pickle.load(open('/Users/alex/Documents/Insight/insight-projects/data/interim/X_train.pickle', 'rb'))
to_scale = [i for i in X_train.columns if i not in
            ['ccn', 'State', 'Ownership Type']]
scaler = StandardScaler()
scaler.fit(df[to_scale])


st.title('Negative Hospice Recommendation Prediction')

#
state_select = st.selectbox(
    'Provider State',
    sorted(df['State'].unique())
)
#
provider = st.selectbox(
    'Select a Provider',
     df[df['State']==state_select]['Facility Name'].to_numpy())
#

# Show performance compared to others
st.subheader('Negative Recommendations: How does this hospice compare?')

comp_type = st.selectbox(
    'Comparison Type',
    ['State', 'National']
)

if comp_type=='State':
    hist_data=[[i for i in df[df['State']==state_select]['neg_rec']]]
    hist_label = state_select
else:
    hist_data=[[i for i in df['neg_rec']]]
    hist_label='National'
distplot = ff.create_distplot(hist_data,
                              [hist_label])

provider_neg_rec = df[df['Facility Name'] == provider]['neg_rec'].to_numpy()

distplot.add_shape(
        # Line Vertical
        go.layout.Shape(
            type="line",
            x0=provider_neg_rec[0],
            y0=0,
            x1=provider_neg_rec[0],
            y1=.2,
            line=dict(
                color="Red",
                width=3
            )
))
st.plotly_chart(distplot)

#%%
# Visualize prediction
st.subheader('Model Predictions')

st.text("""Some factors predict scores more strongly than others. The relative 
magnitude of each predictor in the mode is summarized below.
""")

ranef = pd.read_csv('/Users/alex/Documents/Insight/insight-projects/data/interim/ranef.csv')
fixef = pd.read_csv('/Users/alex/Documents/Insight/insight-projects/data/interim/fixef.csv')
fixef.columns = ['Predictor', 'Coefficient']
st.bar_chart(fixef.set_index('Predictor')[1:])

# Gather predictions
pred = 5

st.subheader('Prediction Comparison')
st.text(f"""On average, hospice facilities with {provider}'s survey scores and 
provider attributes score {pred}. You can explore how facilities with different
predictor values score on average and compare it with your facility by 
adjusting the sliders below. The comparison estimate is in blue.
""")

prov_df = df.loc[df['Facility Name'] == provider, X_train.columns]
comp_df = prov_df.copy()


# Visualize adjustments
RESPECT_BBV = st.slider(
    'RESPECT_BBV',
    df['RESPECT_BBV'].min(),
    df['RESPECT_BBV'].max(),
    prov_df['RESPECT_BBV'].to_numpy()[0]
)
TEAM_COMM_TBV = st.slider(
    'TEAM_COMM_TBV',
    df['TEAM_COMM_TBV'].min(),
    df['TEAM_COMM_TBV'].max(),
    prov_df['TEAM_COMM_TBV'].to_numpy()[0]
)
EMO_REL_BBV = st.slider(
    'EMO_REL_BBV',
    df['EMO_REL_BBV'].min(),
    df['EMO_REL_BBV'].max(),
    prov_df['EMO_REL_BBV'].to_numpy()[0]
)
TIMELY_CARE_BBV = st.slider(
    'TIMELY_CARE_BBV',
    df['TIMELY_CARE_BBV'].min(),
    df['TIMELY_CARE_BBV'].max(),
    prov_df['TIMELY_CARE_BBV'].to_numpy()[0]
)

comp_df['RESPECT_BBV'] = RESPECT_BBV
comp_df['TEAM_COMM_TBV'] = RESPECT_BBV
comp_df['EMO_REL_BBV'] = RESPECT_BBV
comp_df['TIMELY_CARE_BBV'] = RESPECT_BBV





if comp_type=='State':
    hist_data=[[i for i in df[df['State']==state_select]['neg_rec']]]
    hist_label = state_select
else:
    hist_data=[[i for i in df['neg_rec']]]
    hist_label='National'

distplot = ff.create_distplot(hist_data,
                              [hist_label])



# Get preds
reg = LinearRegression()
regvars = ['percRuralZipBen', 'percSOSinpatientHospice',
     'EMO_REL_BBV', 'RESPECT_BBV', 'RESPECT_TBV', 'SYMPTOMS_BBV',
     'TEAM_COMM_BBV', 'TEAM_COMM_MBV', 'TEAM_COMM_TBV', 'TIMELY_CARE_BBV',
     'TIMELY_CARE_TBV', 'TRAINING_BBV', 'distinctBens'
     ]
reg_df = X_train[regvars]
reg.fit(reg_df, y_train)

prov_df_s = scaler.transform(prov_df[to_scale])
prov_df_s = pd.DataFrame(prov_df_s, columns=to_scale)

prov_pred = reg.predict(prov_df_s[regvars])




comp_df = prov_df.copy()
comp_df['RESPECT_BBV'] = RESPECT_BBV
comp_df['TEAM_COMM_TBV'] = TEAM_COMM_TBV
comp_df['EMO_REL_BBV'] = EMO_REL_BBV
comp_df['TIMELY_CARE_BBV'] = TIMELY_CARE_BBV
comp_df_s = scaler.transform(comp_df[to_scale])
comp_df_s = pd.DataFrame(comp_df_s, columns=to_scale)
comp_pred = reg.predict(comp_df_s[regvars])


distplot.add_shape(
        # Line Vertical
        go.layout.Shape(
            type="line",
            x0=prov_pred[0],
            y0=0,
            x1=prov_pred[0],
            y1=.2,
            line=dict(
                color="Red",
                width=3
            )
))

distplot.add_shape(
        # Line Vertical
        go.layout.Shape(
            type="line",
            x0=comp_pred[0],
            y0=0,
            x1=comp_pred[0],
            y1=.2,
            line=dict(
                color="Blue",
                width=3
            )
))

st.plotly_chart(distplot)

