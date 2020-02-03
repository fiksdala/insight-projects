#%% imports

from xgboost import XGBRegressor
from hospice_project.data.transformer import MyScaler
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

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
xgb_y = y_train[mask_train].to_numpy()/100

#%% Adjust for extreme [0,1] values (see R beta regression document)
xgb_y = (xgb_y * (xgb_y.shape[0] - 1) + 0.5) / xgb_y.shape[0]

#%% fit transformer
steps = [('scaler', MyScaler(dont_scale=var_dict['dummy_vars'])),
         ('knn', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(xgb_x)

#%%
xgb_r = XGBRegressor(objective='reg:logistic')
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


gs = RandomizedSearchCV(xgb_r, params)
gs.fit(pipe.transform(xgb_x), xgb_y)

#%%
gs.best_score_

r2_score(xgb_y, gs.predict(pipe.transform(xgb_x)))
