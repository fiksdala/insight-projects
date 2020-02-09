import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_validate
import hospice_project.definitions as defs
import hospice_project.data.transformers as t
from hospice_project.data.train_test_split import get_train_test
import shap

complete_df = pd.read_pickle('data/interim/complete_df.pickle')
X_raw, y = get_train_test(complete_df,
                      defs.sparse_vars,
                      'would_recommend',
                          return_full=True)

X_raw_er, y_er = get_train_test(complete_df,
                      defs.sparse_vars,
                      'RATING_EST',
                                return_full=True)

X_raw.rename(columns=renames, inplace=True)
X_raw_er.rename(columns=renames, inplace=True)

steps = [('scaler', t.MyScaler(dont_scale='for_profit')),
             ('knn', t.KNNKeepDf())]
pipe = Pipeline(steps)
pipe.fit(X_raw, defs.dummy_vars)
X = pipe.transform(X_raw)

#%%
# RandomSearch best XGBRegressor parameters
params = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.5, 1 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

rs = RandomizedSearchCV(XGBRegressor(),
                        params)
rs.fit(X.to_numpy(), y)

#%% Fit models
wr_xgb = XGBRegressor(**rs.best_params_)
wr_xgb.fit(X, y)

# Fit to estimated rate dataframe
pipe.fit(X_raw_er, defs.dummy_vars)
X_er = pipe.transform(X_raw_er)

er_xgb = XGBRegressor(**rs.best_params_)
er_xgb.fit(X_er, y_er)


#%% Get predictions
to_predict = complete_df.loc[complete_df['RECOMMEND_BBV'].isna(),
                             defs.sparse_vars]
pipe.fit(X_raw, defs.dummy_vars)
wr_preds = wr_xgb.predict(pipe.transform(to_predict))

pipe.fit(X_raw_er, defs.dummy_vars)
er_preds = er_xgb.predict(pipe.transform(to_predict))

#%% Make prediction dataframe and save
sparse_preds = pd.DataFrame({
    'ccn': complete_df[complete_df['RECOMMEND_BBV'].isna()]['ccn'],
    'would_recommend': wr_preds,
    'RATING_EST': er_preds
})

sparse_preds.to_pickle('models/sparse_preds.pickle')

#%% Save fitted XGB objects
pd.to_pickle(wr_xgb, 'models/would_recommend_xgb.pickle')
pd.to_pickle(er_xgb, 'models/rating_xgb.pickle')

#%%
renames = {
'percWhite': 'Percent White',
'n_win_30': 'Facilities Within 30 mi',
'nearest': 'Distance to Nearest Facility',
'percRuralZipBen': 'Percent Rural',
'aveNurseMin7dp': 'Nurse Minutes Last Week of Life',
'percSOSskilledNurse': 'Percent Skilled Nursing Service',
'Ownership Type_For-Profit': 'For-Profit',
'homeHealthCtPB': 'Home Aide Visits per Patient',
'aveHCCscore': 'Mean HCC Score',
'nurseVisitCtPB': 'Nurse Visits per Patient',
'Ownership Type_Non-Profit': 'Non-Profit',
'totalMedStandPayPB': 'Medicate Payments per Patient',
'socialWorkCtPB': 'Social Work Visits per Patient',
'percSOSDassisLiv': 'Percent Assisted Living',
'totChargePB': 'Total Charges per Patient',
'epStayCt': 'Stay Count',
'n_win_90': 'Facilities Within 90 mi',
'physicianCtPB': 'Physician Visits per Patient',
'daysService': 'Service Days',
'percWhite': 'Dyspnea Screening',
'percSOSinpatientHospice': 'Percent Inpatient',
'aveSocialWork7dp': 'Social Work Minutes Last Week of Life',
'distinctBens': 'Distinct Beneficiaries',
'days_operation': 'Facility Age',
'percSOSDhome': 'Percent Home Visits',
'percSOSlongtermcare': 'Percent Long-term Care',
'Care_Provided_Inpatient_Hospice_Yes': 'Offers Inpatient Hospice'
}

X.rename(columns=renames, inplace=True)
shap_values = shap.TreeExplainer(wr_xgb).shap_values(X)
shap.summary_plot(shap_values, X)

#%%
X_er.rename(columns=renames, inplace=True)
shap_values_er = shap.TreeExplainer(er_xgb).shap_values(X_er)
shap.summary_plot(shap_values_er, X_er)

#%%
from PIL import Image
im = Image.open('models/would_recommend_shap.png')