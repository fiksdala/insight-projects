import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from hospice_project.data.transformer import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier



X_train = pd.read_pickle('data/processed/X_train.pickle')
service_quality = [i for i in X_train.columns
 if ('BBV' in i) | ('MBV' in i) | ('TBV' in i)]
X = X_train.loc[~X_train['RECOMMEND_BBV'].isna()].reset_index(drop=True).copy()
y = X_train['RECOMMEND_BBV'][~X_train['RECOMMEND_BBV'].isna()].to_numpy()
var_dict = pd.read_pickle('data/interim/var_dict.pickle')

feature_groups = {
    'Service Quality': ['EMO_REL_BTR', 'RESPECT_BTR', 'SYMPTOMS_BTR',
                        'TEAM_COMM_BTR', 'TIMELY_CARE_BTR', 'TRAINING_BTR'],
    'Services Delivered': ['H_009_01_OBSERVED', 'percSOSDassisLiv',
                           'nurseVisitCtPB', 'socialWorkCtPB',
                           'physicianCtPB'],
    'Services Offered': ['Care_Provided_Assisted_Living_Yes',
                         'Care_Provided_Home_Yes',
                         'Care_Provided_Inpatient_Hospice_Yes',
                         'Care_Provided_Skilled_Nursing_Yes'],
    'Patients Served': ['percBlack', 'percHisp', 'percBen30orFewerDays'],
    'Financial Details': 'totalMedStandPayPB'
}

lr_vars = ['EMO_REL_BTR', 'RESPECT_BTR', 'SYMPTOMS_BTR',
                        'TEAM_COMM_BTR', 'TIMELY_CARE_BTR', 'TRAINING_BTR',
           'H_009_01_OBSERVED', 'percSOSDassisLiv',
           'nurseVisitCtPB', 'socialWorkCtPB',
           'physicianCtPB', 'Care_Provided_Assisted_Living_Yes',
                         'Care_Provided_Home_Yes',
                         'Care_Provided_Inpatient_Hospice_Yes',
                         'Care_Provided_Skilled_Nursing_Yes',
'percBlack', 'percHisp', 'percBen30orFewerDays', 'totalMedStandPayPB'
           ]


#%%

actionable_vars = ['EMO_REL_BTR', 'RESPECT_BTR', 'SYMPTOMS_BTR',
                   'TEAM_COMM_BTR', 'TIMELY_CARE_BTR', 'TRAINING_BTR',
                   'H_009_01_OBSERVED', 'percSOSDassisLiv',
                   'nurseVisitCtPB', 'socialWorkCtPB',
                   'physicianCtPB',
                   'percBen30orFewerDays']

steps = [('scaler', MyScaler(var_dict['dummy_vars'])),
         ('knn', KNNKeepDf())]
pipe = Pipeline(steps)
pipe.fit(X[actionable_vars])

rfc_scores = cross_validate(RandomForestClassifier(max_depth=5),
                                         pipe.transform(X[actionable_vars]),
                                         y,
                                         cv=RepeatedKFold(random_state=42),
                                         scoring=('r2',
                                                  'neg_mean_squared_error',
                                                  'neg_mean_absolute_error'),
                                         return_train_score=True
                                         )

#%%
print(
knn_scores['test_neg_mean_squared_error'].mean(),
act_vars_k['test_neg_mean_squared_error'].mean(),
rfc_scores['test_neg_mean_squared_error'].mean()
)