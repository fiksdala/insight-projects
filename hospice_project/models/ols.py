import pandas as pd
import numpy as np
import pymc3 as pm
from matplotlib import pyplot as plt
from hospice_project.data.transformer import MyScaler
from hospice_project.models.bayes_vis import test_model
from sklearn.pipeline import Pipeline
import seaborn as sns
from theano import shared
import statsmodels.api as sm

var_dict = pd.read_pickle('data/interim/var_dict.pickle')


r_keep_vars = ["H_009_01_OBSERVED",
"EMO_REL_BTR",
"RESPECT_BTR",
"SYMPTOMS_BTR",
"TEAM_COMM_BTR",
"TIMELY_CARE_BTR",
"TRAINING_BTR",
"percBlack",
"percHisp",
"percBen30orFewerDays",
"percSOSDassisLiv",
"nurseVisitCtPB",
"socialWorkCtPB",
"physicianCtPB",
"totalMedStandPayPB",
"Care_Provided_Assisted_Living_Yes",
"Care_Provided_Home_Yes",
"Care_Provided_Inpatient_Hospice_Yes",
"Care_Provided_Skilled_Nursing_Yes"]

X_train = pd.read_pickle('data/processed/X_train.pickle')

#%% Loop DF: drop recommend vars and make recommend_bbv Y
ols_x = X_train[r_keep_vars][~X_train['RECOMMEND_BBV'].isna()]
ols_y = X_train['RECOMMEND_BBV'][~X_train['RECOMMEND_BBV'].isna()].reset_index(drop=True)
ols_y = [0.0001 if i == 0 else i/100 for i in ols_y]
# set up scale/impute pipeline
steps = [('scaler', MyScaler(dummy_vars=var_dict['dummy_vars'])),
         ('KNNImputer', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(bayes_x)
X = pipe.transform(ols_x)
X = pd.DataFrame(X, columns=ols_x.columns)

#%%
olsM = sm.OLS(ols_y, X).fit()
olsM.bse* np.sqrt(X.shape[0])

#%%
ols_y