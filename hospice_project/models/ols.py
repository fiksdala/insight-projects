import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

#%%
var_dict = pd.read_pickle('data/interim/var_dict.pickle')

class MyScaler(BaseEstimator, TransformerMixin):
    def __init__(self, dont_scale=None):
        self.dont_scale = dont_scale
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        if self.dont_scale is not None:
            self.scaler.fit(X[[i for i in X.columns
                               if i not in self.dont_scale]])
            self.colnames_ = [i for i in X.columns
                              if i not in self.dont_scale] + [
                i for i in X.columns if i in self.dont_scale
            ]
        else:
            self.scaler.fit(X)
            self.colnames_ = X.columns
        return self

    def transform(self, X, y=None, **fit_params):
        if self.dont_scale is not None:
            X_scaled = self.scaler.transform(
                X[[i for i in X.columns if i not in self.dont_scale]]
            )
            output = np.concatenate([X_scaled,
                                     X[[i for i in X.columns
                                        if i in self.dont_scale]]
                                     ], axis=1)
        else:
            output = self.scaler.transform(X)

        return pd.DataFrame(output,
                            columns=self.colnames_)

    def inverse_transform(self, X, y=None):
        if self.dont_scale is not None:
            X_inverse = self.scaler.inverse_transform(
                X[[i for i in X.columns if i not in self.dont_scale]]
            )
            output = np.concatenate([X_inverse,
                                     X[[i for i in X.columns
                                        if i in self.dont_scale]]
                                     ], axis=1)
        else:
            output = self.scaler.inverse_transform(X)
        return output



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

X_train = pd.read_pickle('data/processed/X_test.pickle')

#%% ols_dfs drop recommend vars and make recommend_bbv Y
ols_x = X_train[r_keep_vars][~X_train['RECOMMEND_BBV'].isna()]
ols_y = X_train['RECOMMEND_BBV'][~X_train['RECOMMEND_BBV'].isna()].reset_index(drop=True)

# set up scale/impute pipeline
steps = [('scaler', MyScaler(dont_scale=var_dict['dummy_vars'])),
         ('KNNImputer', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(ols_x)
X = pipe.transform(ols_x)
X = pd.DataFrame(X, columns=ols_x.columns)

#%%
olsM = sm.OLS(ols_y, X).fit()

#%%
preds = olsM.get_prediction(X)
preds.summary_frame()

#%% Dict
ols_dict = {'pipe': pipe,
            'olsM': olsM,
            'keep_vars': r_keep_vars}
pickle.dump(olsM, open('data/interim/ols_obj.pickle', 'wb'))
pickle.dump(pipe, open('data/interim/pipe.pickle', 'wb'))

#%%
pipe['scaler'].inverse_transform(X)

#%%
olsM.summary()


#%%
a = np.array(1,2,3)


