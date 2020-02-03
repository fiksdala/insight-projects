import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from hospice_project.data.transformer import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate



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

steps = [('scaler', MyScaler(var_dict['dummy_vars'])),
         ('knn', KNNKeepDf())]
pipe = Pipeline(steps)
#%%
pipe.fit(X[lr_vars])
lr = LinearRegression()
lr.fit(pipe.transform(X[lr_vars]), y)

#%%
explore_df = pipe.transform(X[lr_vars])
explore_df['y'] = y
explore_df['pred'] = lr.predict(pipe.transform(X[lr_vars]))

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
pipe.fit(X[lr_vars])

all_vars_k = cross_validate(LinearRegression(),
                                         pipe.transform(X[lr_vars]),
                                         y,
                                         cv=RepeatedKFold(random_state=42),
                                         scoring=('r2',
                                                  'neg_mean_squared_error',
                                                  'neg_mean_absolute_error'),
                                         return_train_score=True
                                         )
steps = [('scaler', MyScaler(var_dict['dummy_vars'])),
         ('knn', KNNKeepDf())]
pipe_act = Pipeline(steps)
pipe_act.fit(X[actionable_vars])
act_vars_k = cross_validate(LinearRegression(),
                                         pipe_act.transform(X[actionable_vars]),
                                         y,
                                         cv=RepeatedKFold(random_state=42),
                                         scoring=('r2',
                                                  'neg_mean_squared_error',
                                                  'neg_mean_absolute_error'),
                                         return_train_score=True
                                         )

#%%
print(
np.mean([np.sqrt(abs(i)) for i in all_vars_k['test_neg_mean_squared_error']]),
np.mean([np.sqrt(abs(i)) for i in act_vars_k['test_neg_mean_squared_error']])
)

#%%
initial_df = pd.read_pickle('data/interim/initial_df.pickle')
X_all = initial_df[~initial_df['RECOMMEND_BBV'].isna()].reset_index(
    drop=True).copy()
y_all = X_all['RECOMMEND_BBV'].to_numpy()
steps_act = [('scaler', MyScaler(var_dict['dummy_vars'])),
         ('knn', KNNKeepDf())]
pipe_act = Pipeline(steps_act)
pipe_act.fit(X_all[actionable_vars])
lr_act = LinearRegression()
lr_act.fit(pipe_act.transform(X_all[actionable_vars]), y_all)

#%%
coef_df = pd.DataFrame(lr_act.coef_.reshape(1,-1),
             columns=actionable_vars)


#%%

# to_r = pipe_act.transform(X_all[actionable_vars])
# y_all = y_all/100
# to_r['y'] = (y_all * (y_all.shape[0] - 1) + 0.5) / y_all.shape[0]
# to_r.to_csv('data/processed/final_r.csv', index=False)

#%%
full_df = pipe_act.transform(X_all[actionable_vars])


#%%
test_sample = full_df.sample(1)
#%%
def recommender(obs):
    potential_improvement = []
    for var in obs:
        obs_pos = obs[var].to_numpy() > 0
        coef_pos = coef_df[var].to_numpy() > 0
        obs_neg = obs[var].to_numpy() < 0
        coef_neg = coef_df[var].to_numpy() < 0

        if obs_pos and coef_pos:
            potential_improvement.append(var)
        if obs_neg and coef_neg:
            potential_improvement.append(var)
    output = obs[potential_improvement].transpose()
    output.columns = ['obs']
    # return sorted by absolute values
    return output.iloc[(-np.abs(output['obs'].values)).argsort()]

recommender(test_sample)