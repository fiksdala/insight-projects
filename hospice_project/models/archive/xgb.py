#%% Imports
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from hospice_project.data.transformers import MyScaler
from sklearn.pipeline import Pipeline
import numpy as np

var_dict = pd.read_pickle('data/interim/var_dict.pickle')


r_keep_vars = ["H_002_09_OBSERVED",
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

# Loop DF: drop recommend vars and make recommend_bbv Y
xgb_x = X_train.loc[~X_train['RECOMMEND_BBV'].isna(),
                    r_keep_vars].reset_index(drop=True).copy()
xgb_y = X_train.loc[~X_train['RECOMMEND_BBV'].isna(),
                    'RECOMMEND_BBV'].reset_index(drop=True).copy()

# set up scale/impute pipeline
steps = [('scaler', MyScaler(dummy_vars=var_dict['dummy_vars'])),
         ('KNNImputer', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(xgb_x)
#%%
xgb = XGBClassifier(objective='reg:squarederror', min_child_weight=5,
                    n_estimators=100, max_depth=5)
# xgb.fit(pipe.transform(xgb_x), xgb_y)

kfold = KFold(n_splits=5)
results = cross_val_score(xgb, pipe.transform(xgb_x), xgb_y, cv=kfold,
                          scoring='neg_root_mean_squared_error')


#%%
print(results)
xgb.fit(pipe.transform(xgb_x), xgb_y)
np.sqrt(mean_squared_error(xgb_y,
                   xgb.predict(pipe.transform(xgb_x))))

#%%
results

