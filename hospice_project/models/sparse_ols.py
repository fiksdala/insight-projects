import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import numpy as np

sparse_keep_vars = pd.read_pickle('data/interim/sparse_keep_vars.pickle')

X_train = pd.read_pickle('data/processed/X_train.pickle')

#%% ols_dfs drop recommend vars and make recommend_bbv Y
ols_x = X_train[sparse_keep_vars][~X_train['RECOMMEND_BBV'].isna()]
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
olsM.summary()

#%%
pd.to_pickle(olsM, 'data/interim/sparse_ols.pickle')
