from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from hospice_project.data.transformer import *
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot as plt

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

X_train = pd.read_pickle('data/processed/X_train.pickle')
service_quality = [i for i in X_train.columns
 if ('BBV' in i) | ('MBV' in i) | ('TBV' in i)]
X = X_train.loc[~X_train['RECOMMEND_BBV'].isna()].reset_index(drop=True).copy()
y = X_train['RECOMMEND_BBV'][~X_train['RECOMMEND_BBV'].isna()].to_numpy()

#%% service quality raw vs. bottom-to-top ratio vs. PCA 1st component
# raw
sq_kfold_results = {}
sq_raw = [i for i in service_quality
          if i not in ['RATING_BBV', 'RATING_TBV',
                       'RECOMMEND_BBV', 'RECOMMEND_TBV']]
steps = [('scaler', MyScaler()),
         ('KNNImputer', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(X[sq_raw])

#%%
sq_kfold_results['raw'] = cross_validate(LinearRegression(),
                                         pipe.transform(X[sq_raw]),
                                         y,
                                         cv=RepeatedKFold(random_state=42),
                                         scoring=('r2',
                                                  'neg_mean_squared_error',
                                                  'neg_mean_absolute_error'),
                                         return_train_score=True

                                         )

#%%
ratio_vars = ['EMO_REL_BTR', 'RESPECT_BTR', 'SYMPTOMS_BTR',
              'TEAM_COMM_BTR', 'TIMELY_CARE_BTR', 'TRAINING_BTR']

steps = [('scaler', MyScaler()),
         ('KNNImputer', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(X[ratio_vars])

sq_kfold_results['ratio'] = cross_validate(LinearRegression(),
                                         pipe.transform(X[ratio_vars]),
                                         y,
                                         cv=RepeatedKFold(random_state=42),
                                         scoring=('r2',
                                                  'neg_mean_squared_error',
                                                  'neg_mean_absolute_error'),
                                         return_train_score=True
                                         )

#%%
steps = [('scaler', MyScaler()),
         ('knn', KNNKeepDf()),
         ('pca', GetComponents({'sq_raw': sq_raw}))]
pipe = Pipeline(steps)
pipe.fit(X[sq_raw])

sq_kfold_results['PCA_raw'] = cross_validate(LinearRegression(),
                                         pipe.transform(X[sq_raw]),
                                         y,
                                         cv=RepeatedKFold(random_state=42),
                                         scoring=('r2',
                                                  'neg_mean_squared_error',
                                                  'neg_mean_absolute_error'),
                                         return_train_score=True
                                         )

steps = [('scaler', MyScaler()),
         ('knn', KNNKeepDf()),
         ('pca', GetComponents({'ratio_vars': ratio_vars}))]
pipe = Pipeline(steps)
pipe.fit(X[ratio_vars])
sq_kfold_results['PCA_ratio'] = cross_validate(LinearRegression(),
                                         pipe.transform(X[ratio_vars]),
                                         y,
                                         cv=RepeatedKFold(random_state=42),
                                         scoring=('r2',
                                                  'neg_mean_squared_error',
                                                  'neg_mean_absolute_error'),
                                         return_train_score=True
                                         )

#%%
for var_group in sq_kfold_results.keys():
    print(var_group)
    print([sq_kfold_results[var_group][i].mean()
           for i in ['test_r2', 'test_neg_mean_squared_error',
                     'test_neg_mean_absolute_error']]

          )

for var_group in sq_kfold_results.keys():
    print(var_group)
    print(
        np.mean(
            np.sqrt(
                abs(sq_kfold_results[var_group]['test_neg_mean_squared_error']
                    )
            )
        )
    )

