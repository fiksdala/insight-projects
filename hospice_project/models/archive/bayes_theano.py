#%% Imports
import pandas as pd
import numpy as np
import pymc3 as pm
from matplotlib import pyplot as plt
from hospice_project.data.transformers import MyScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
from theano import shared
from scipy import stats
import pickle

#%% Data frames and variable lists
X_train = pd.read_pickle('data/processed/X_train.pickle')
var_dict = pickle.load(open('data/interim/var_dict.pickle', 'rb'))
r_keep = pickle.load(open('data/interim/r_keep.pickle', 'rb'))


# Drop dv and obs missing RECOMMEND_BBV
bayes_x = X_train[r_keep][~X_train['RECOMMEND_BBV'].isna()]

# Assign y var
bayes_y = X_train['RECOMMEND_BBV'][~X_train['RECOMMEND_BBV'].isna()]
bayes_y.reset_index(drop=True, inplace=True)

# Scale and shift, make percentages proportions and shift by .01 to avoid 0s
# This will ensure that we can estimate beta likelihood and posterior
bayes_y = (bayes_y / 100) + .5

# Scale and impute using pipeline
steps = [('scaler', MyScaler(dont_scale=var_dict['dummy_vars'])),
         ('KNNImputer', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(bayes_x)
X = pipe.transform(bayes_x)
X = pd.DataFrame(X, columns=bayes_x.columns)

#%% Build Model

# Make vars shared so predictions can be made later

bayes_y_s = shared(bayes_y.to_numpy())

# Use loop to make dict of shared predictors
shared_dict = {}
for var in X.columns:
    shared_dict[var] = shared(X[var].to_numpy())

#%%
# Specify model
with pm.Model() as model:
    # Intercept
    b_0 = pm.Normal('b_0', mu=.5, sd=.03)
    b_1 = pm.Normal("b_1", mu=0.02, sd=.01)
    b_2 = pm.Normal("b_2", mu=0, sd=.01)
    # b_3 = pm.Normal("b_3", mu=0.009, sd=.03)
    # b_4 = pm.Normal("b_4", mu=0.002, sd=.03)
    # b_5 = pm.Normal("b_5", mu=0.005, sd=.03)
    # b_6 = pm.Normal("b_6", mu=0.003, sd=.03)
    # b_7 = pm.Normal("b_7", mu=0.003, sd=.03)
    # b_8 = pm.Normal("b_8", mu=0.001, sd=.03)
    # b_9 = pm.Normal("b_9", mu=-0.0, sd=.03)
    # b_10 = pm.Normal("b_10", mu=-0.007, sd=.03)
    # b_11 = pm.Normal("b_11", mu=0.003, sd=.03)
    # b_12 = pm.Normal("b_12", mu=-0.0, sd=.03)
    # b_13 = pm.Normal("b_13", mu=-0.0, sd=.03)
    # b_14 = pm.Normal("b_14", mu=-0.002, sd=.03)
    # b_15 = pm.Normal("b_15", mu=-0.004, sd=.03)
    # b_16 = pm.Normal("b_16", mu=-0.002, sd=.03)
    # b_17 = pm.Normal("b_17", mu=0.044, sd=.03)
    # b_18 = pm.Normal("b_18", mu=0.004, sd=.03)
    # b_19 = pm.Normal("b_19", mu=-0.001, sd=.03)

    # Model error
    model_err = pm.HalfNormal('model_err', sd=.01)

    # Expected value
    y_est = pm.math.invlogit(b_0 +
                             b_1 * shared_dict["H_009_01_OBSERVED"] +
                             b_2 * shared_dict["EMO_REL_BTR"]
                             # b_3 * shared_dict["RESPECT_BTR"] +
                             # b_4 * shared_dict["SYMPTOMS_BTR"] +
                             # b_5 * shared_dict["TEAM_COMM_BTR"] +
                             # b_6 * shared_dict["TIMELY_CARE_BTR"] +
                             # b_7 * shared_dict["TRAINING_BTR"]
                             # b_8 * shared_dict["percBlack"] +
                             # b_9 * shared_dict["percHisp"] +
                             # b_10 * shared_dict["percBen30orFewerDays"] +
                             # b_11 * shared_dict["percSOSDassisLiv"] +
                             # b_12 * shared_dict["nurseVisitCtPB"] +
                             # b_13 * shared_dict["socialWorkCtPB"] +
                             # b_14 * shared_dict["physicianCtPB"] +
                             # b_15 * shared_dict["totalMedStandPayPB"] +
                             # b_16 * shared_dict["Care_Provided_Assisted_Living_Yes"] +
                             # b_17 * shared_dict["Care_Provided_Home_Yes"] +
                             # b_18 * shared_dict["Care_Provided_Inpatient_Hospice_Yes"] +
                             # b_19 * shared_dict["Care_Provided_Skilled_Nursing_Yes"]
                             )
    # Data likelihood
    y_like = pm.Beta('y_like', mu=y_est, sd=model_err,
                     observed=bayes_y_s)
    
    # Trace
    # normal_trace = pm.sample(tune=1000)

with model:
    start = pm.find_MAP()
    hierarchical_trace = pm.sample(2000, step=pm.Metropolis(), start=start, tune=1000)

#%%

hierarchical_trace['b_1']
#%%

i = 1
for k in shared_dict.keys():
    print('b_'+str(i)+' * '+ 'shared_dict["'+k+'"]' + ' + ')
    i += 1

#%%
for i in np.arange(1,20):
    print('b_'+str(i)+ ' = pm.Normal("b_'+str(i)+'", mu='+
          str(round(olsM.params[i-1], 3))+', sd=.03)')




#%%
with pm.Model() as model:
    # Intercept
    b_0 = pm.Normal('b_0', mu=.05, sd=.03)
    b_1 = pm.Normal("b_1", mu=0.02, sd=.01)
    b_2 = pm.Normal("b_2", mu=0, sd=.001)
    # b_3 = pm.Normal("b_3", mu=0.009, sd=.03)
    # b_4 = pm.Normal("b_4", mu=0.002, sd=.03)
    # b_5 = pm.Normal("b_5", mu=0.005, sd=.03)
    # b_6 = pm.Normal("b_6", mu=0.003, sd=.03)
    # b_7 = pm.Normal("b_7", mu=0.003, sd=.03)
    # b_8 = pm.Normal("b_8", mu=0.001, sd=.03)
    # b_9 = pm.Normal("b_9", mu=-0.0, sd=.03)
    # b_10 = pm.Normal("b_10", mu=-0.007, sd=.03)
    # b_11 = pm.Normal("b_11", mu=0.003, sd=.03)
    # b_12 = pm.Normal("b_12", mu=-0.0, sd=.03)
    # b_13 = pm.Normal("b_13", mu=-0.0, sd=.03)
    # b_14 = pm.Normal("b_14", mu=-0.002, sd=.03)
    # b_15 = pm.Normal("b_15", mu=-0.004, sd=.03)
    # b_16 = pm.Normal("b_16", mu=-0.002, sd=.03)
    # b_17 = pm.Normal("b_17", mu=0.044, sd=.03)
    # b_18 = pm.Normal("b_18", mu=0.004, sd=.03)
    # b_19 = pm.Normal("b_19", mu=-0.001, sd=.03)

    # Model error
    model_err = pm.HalfNormal('model_err', sd=.01)

    # Expected value
    y_est = pm.math.invlogit(b_0 +
                             b_1 * shared_dict["H_009_01_OBSERVED"] +
                             b_2 * shared_dict["EMO_REL_BTR"]
                             # b_3 * shared_dict["RESPECT_BTR"] +
                             # b_4 * shared_dict["SYMPTOMS_BTR"] +
                             # b_5 * shared_dict["TEAM_COMM_BTR"] +
                             # b_6 * shared_dict["TIMELY_CARE_BTR"] +
                             # b_7 * shared_dict["TRAINING_BTR"] +
                             # b_8 * shared_dict["percBlack"] +
                             # b_9 * shared_dict["percHisp"] +
                             # b_10 * shared_dict["percBen30orFewerDays"] +
                             # b_11 * shared_dict["percSOSDassisLiv"] +
                             # b_12 * shared_dict["nurseVisitCtPB"] +
                             # b_13 * shared_dict["socialWorkCtPB"] +
                             # b_14 * shared_dict["physicianCtPB"] +
                             # b_15 * shared_dict["totalMedStandPayPB"] +
                             # b_16 * shared_dict["Care_Provided_Assisted_Living_Yes"] +
                             # b_17 * shared_dict["Care_Provided_Home_Yes"] +
                             # b_18 * shared_dict["Care_Provided_Inpatient_Hospice_Yes"] +
                             # b_19 * shared_dict["Care_Provided_Skilled_Nursing_Yes"]
                             )
    # Data likelihood
    y_like = pm.Beta('y_like', mu=y_est, sd=model_err,
                     observed=bayes_y_s)

    # Trace
    normal_trace = pm.sample(tune=1000)

