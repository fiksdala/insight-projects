import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from scipy.special import logit, expit

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
X_test = pd.read_pickle('data/processed/X_test.pickle')
#%% ols_dfs drop recommend vars and make recommend_bbv Y
ols_x = X_train[r_keep_vars][~X_train['RECOMMEND_BBV'].isna()]
ols_y = X_train['RECOMMEND_BBV'][~X_train['RECOMMEND_BBV'].isna()].reset_index(drop=True)

ols_x_test = X_test[r_keep_vars][~X_test['RECOMMEND_BBV'].isna()]
ols_y_test = X_test['RECOMMEND_BBV'][~X_test['RECOMMEND_BBV'].isna()].reset_index(drop=True)


#%%
# set up scale/impute pipeline
steps = [('scaler', MyScaler(dont_scale=var_dict['dummy_vars'])),
         ('KNNImputer', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(ols_x)
X_train_scaled = pipe.transform(ols_x)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=ols_x.columns)

X_test_scaled = pipe.transform(ols_x_test)
# X_test_scaled = pd.DataFrame(X_test_scaled, columns=ols_x.columns)


lr = LinearRegression()

#%%
X_train_scaled.shape
ols_y.shape
#%%

lr.fit(X_train_scaled, ols_y)

#%%
rkf = RepeatedKFold()

#%%

lr_full = cross_validate(LinearRegression(),
                         X_train_scaled,
                         xgb_y,
                         cv=rkf,
                        scoring=('r2', 'neg_mean_squared_error',
                                 'neg_mean_absolute_error'),
                        return_train_score=True
                                 )
#%%

lr_full

#%%
np.mean([abs(i) for i in
    lr_full['test_neg_mean_absolute_error']])

#%%
np.mean([np.sqrt(abs(i)) for i in
    lr_full['test_neg_mean_squared_error']])

#%%
np.mean(lr_full['test_r2'])

#%%
np.mean(lr_full['test_r2'])

#%%
lr_pcs = cross_validate(lr, PCs, xgb_y,
                        scoring=('r2', 'neg_mean_squared_error',
                                 'neg_mean_absolute_error'),
                        return_train_score=True
                                 )

#%%


#%%
np.mean([np.sqrt(abs(i)) for i in
    lr_full['test_neg_mean_squared_error']])

#%%
def expit_mse(y_true, y_pred):
    return mean_squared_error(expit(y_true), expit(y_pred))

def expit_r2(y_true, y_pred):
    return r2_score(expit(y_true), expit(y_pred))

def expit_mae(y_true, y_pred):
    return mean_absolute_error(expit(y_true), expit(y_pred))

custom_score_dict = {
    'neg_mean_squared_error': 'neg_mean_squared_error',
    'expit_mse': make_scorer(expit_mse,greater_is_better=False),
    'expit_r2': make_scorer(expit_r2),
    'expit_mae': make_scorer(expit_mae, greater_is_better=False)
}

lr_full = cross_validate(lr, X_train_scaled, logit(xgb_y), cv=rkf,
                        scoring=custom_score_dict,
                         return_train_score=True)


#%%
np.mean([np.sqrt(abs(i)) for i in lr_full['test_expit_mse']])
