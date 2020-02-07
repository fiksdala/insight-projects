#%% imports

from xgboost import XGBRegressor
from hospice_project.data.transformers import MyScaler
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from hyperopt import hp
from sklearn.linear_model import LinearRegression

#%%
X_train = pd.read_pickle('data/processed/X_train.pickle')
X_test = pd.read_pickle('data/processed/X_test.pickle')
y_train = pd.read_pickle('data/processed/y_train.pickle')
y_test = pd.read_pickle('data/processed/y_test.pickle')
var_dict = pd.read_pickle('data/interim/var_dict.pickle')
keep_vars = pd.read_pickle('data/interim/r_keep.pickle')

mask_train = ~X_train['RECOMMEND_BBV'].isna()
mask_test = ~X_test['RECOMMEND_BBV'].isna()

#%% make xgb x/y (and make y [0,1]
xgb_x = X_train.loc[mask_train, keep_vars].copy()
xgb_y = y_train[mask_train].to_numpy()

#%% Adjust for extreme [0,1] values (see R beta regression document)
xgb_y = (xgb_y * (xgb_y.shape[0] - 1) + 0.5) / xgb_y.shape[0]

#%% fit transformer
steps = [('scaler', MyScaler(dont_scale=var_dict['dummy_vars'])),
         ('knn', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(xgb_x)

#%%
xgb_r = XGBRegressor(loss='reg:logistic')
params = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }
# scores = cross_validate(xgb_r,
#                          pipe.transform(xgb_x),
#                          xgb_y,
#                          scoring=('r2', 'neg_mean_squared_error',
#                                   'neg_mean_absolute_error'),
#                         return_train_score=True)


gs = RandomizedSearchCV(xgb_r, params,
                        scoring='r2'
                        )
gs.fit(pipe.transform(xgb_x), xgb_y)

gs.best_params_

#%%
gs.best_score_

#%%
params = {
    'max_bin': [10,20,50,100,255],
    'num_leaves': [5,10,31,50],
    'min_data_in_leaf': [10,20,30],
    'bagging_fraction': [.1,.3,.5,.7,1]
}

lgb_q = LGBMRegressor(objective='quantile')

gs = RandomizedSearchCV(lgb_q, params,
                        scoring=['r2', 'neg_mean_squared_error',
                                 'neg_mean_absolute_error'],
                        refit='neg_mean_squared_error'
                        )
gs.fit(pipe.transform(xgb_x), xgb_y)

#%%
pd.to_pickle(gs, 'data/interim/lgbm_perf.pickle')

#%%
np.sqrt(-gs.cv_results_['mean_test_neg_mean_squared_error'].mean())
gs.cv_results_['mean_test_r2'].mean()


#%%
linreg_perf = cross_validate(LinearRegression(), pipe.transform(xgb_x), xgb_y,
                        scoring=('r2', 'neg_mean_squared_error',
                                 'neg_mean_absolute_error'),
                        return_train_score=True
                                 )

#%%
linreg_perf
pd.to_pickle(linreg_perf, 'data/interim/linreg_perf.pickle')

#%%
print(linreg_perf['test_neg_mean_absolute_error'].mean(),
      linreg_perf['test_r2'].mean(),
      np.sqrt(-linreg_perf['test_neg_mean_squared_error'].mean()))

#%%
list(hp.quniform('num_leaves', 30, 150, 1))