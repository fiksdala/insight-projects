import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import ShuffleSplit
import copy
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from hospice_project.data.transformers import *
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
initial_df = pd.read_pickle('data/interim/initial_df.pickle')
keep_vars = pd.read_pickle('data/processed/keep_vars.pickle')
sparse_keep_vars = pd.read_pickle('data/interim/sparse_keep_vars.pickle')
var_dict = pd.read_pickle('data/interim/var_dict.pickle')


#%%
sim_miss_df_full = initial_df[sparse_keep_vars].copy()
complete_df = sim_miss_df_full[~initial_df['RECOMMEND_BBV'].isna()].reset_index(drop=True)
y_complete = initial_df[~initial_df['RECOMMEND_BBV'].isna()]['RECOMMEND_BBV'].reset_index(drop=True)
sparse_df = sim_miss_df_full[initial_df['RECOMMEND_BBV'].isna()].reset_index(drop=True)
y_sparse = initial_df[initial_df['RECOMMEND_BBV'].isna()]['RECOMMEND_BBV'].reset_index(drop=True)


#%%
def sim_miss(complete_df, sparse_df):
    """Simulate missing data. Assumes MCAR which may not be valid, so use
    caution with inference"""
    # get missing percentages of each column
    missing_perc = sparse_df.isna().sum()/sparse_df.shape[0]

    # apply missingness percentages of each columns randomly
    for mp in missing_perc.index:
        vals = np.random.uniform(0, 1, size=complete_df.shape[0])
        mask = vals < missing_perc[mp]
        complete_df.loc[mask, mp] = np.nan

    # return complete df with simulated missingness
    return complete_df

#%%######
def resample_kfold(X, y, sparse_df, model, folds=5):
    folds = ShuffleSplit(n_splits=folds, test_size=(1 / folds))
    folds.get_n_splits(X, y)

    model_history = []
    train_r2 = []
    train_mse = []
    train_mae = []

    test_r2 = []
    test_mse = []
    test_mae = []
    for train_index, test_index in folds.split(X, y):
        xtrain, xtest = X.iloc[train_index,:], X.iloc[test_index,:]
        ytrain, ytest = y[train_index], y[test_index]

        # Simulate missingness
        xtest = pd.DataFrame(xtest, columns=X.columns)
        xtest = sim_miss(xtest, sparse_df)

        # Scale/transform xtrain
        steps = [('scaler', MyScaler(var_dict['dummy_vars'])),
                 ('knn', KNNKeepDf())]
        pipe = Pipeline(steps)
        pipe.fit(xtrain)
        xtrain = pipe.transform(xtrain)

        # scale/impute test (with simulated missing)
        xtest = pipe.transform(xtest)

        # Run the model
        loop_model = copy.copy(model)
        loop_model.fit(xtrain, ytrain)

        # Save models
        model_history.append(loop_model)

        # Save Performance
        train_r2.append(r2_score(ytrain, loop_model.predict(xtrain)))
        train_mse.append(mean_squared_error(ytrain, loop_model.predict(xtrain)))
        train_mae.append(mean_absolute_error(ytrain, loop_model.predict(xtrain)))

        test_r2.append(r2_score(ytest, loop_model.predict(xtest)))
        test_mse.append(mean_squared_error(ytest, loop_model.predict(xtest)))
        test_mae.append(mean_absolute_error(ytest, loop_model.predict(xtest)))

    performance = {
        'train': {'r2': train_r2,
                  'mse': train_mse,
                  'mae': train_mae},
        'test':  {'r2': test_r2,
                  'mse': test_mse,
                  'mae': test_mae}
    }
    return model_history, performance

#%%
lr_mhist, lr_perf = resample_kfold(complete_df, y_complete,
                                   sparse_df, LinearRegression())

#%%
knn_mhist, knn_perf = resample_kfold(complete_df, y_complete,
                                   sparse_df, KNeighborsRegressor())


#%%
knn_mhist_10, knn_perf_10 = resample_kfold(complete_df, y_complete,
                                   sparse_df, KNeighborsRegressor(
        n_neighbors=10
    ))

knn_mhist_3, knn_perf_3 = resample_kfold(complete_df, y_complete,
                                   sparse_df, KNeighborsRegressor(
        n_neighbors=3
    ))

knn_mhist_7, knn_perf_7 = resample_kfold(complete_df, y_complete,
                                   sparse_df, KNeighborsRegressor(
        n_neighbors=7
    ))

knn_mhist_15, knn_perf_15 = resample_kfold(complete_df, y_complete,
                                   sparse_df, KNeighborsRegressor(
        n_neighbors=15
    ))

#%%
knn_mhist_30, knn_perf_30 = resample_kfold(complete_df, y_complete,
                                   sparse_df, KNeighborsRegressor(
        n_neighbors=30
    ))


#%%
knn_perf_30

#%%
# RandomSearch best XGBRegressor parameters
params = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.5, 1 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

rs = RandomizedSearchCV(XGBRegressor(),
                        params)
rs.fit(complete_df, y_complete)

#%%
rs.best_params_

#%%
#%%
xgb_mhist, xgb_perf = resample_kfold(complete_df, y_complete,
                                     sparse_df,
                                     XGBRegressor(**rs.best_params_))

#%%
[np.sqrt(i) for i in xgb_perf['test']['mse']]
xgb_perf

#%% Compare knn(30) vs. xgb vs. mean
mean_mae = mean_absolute_error(y_complete,
                              [y_complete.mean() for i in
                               y_complete])
mean_mse = mean_squared_error(y_complete,
                              [y_complete.mean() for i in
                               y_complete])

#%%
# Repeated 5fold (x10)
rkf_knn = []
rkf_xgb = []
for i in range(10):
    l_xgb_mhist, l_xgb_perf = resample_kfold(
        complete_df, y_complete, sparse_df,
        XGBRegressor(**rs.best_params_))
    l_knn_mhist, l_knn_perf = resample_kfold(
        complete_df, y_complete, sparse_df,
        KNeighborsRegressor(n_neighbors=30))

    rkf_xgb.append(l_xgb_perf)
    rkf_knn.append(l_knn_perf)

#%%
rkf_knn_test_mse_mean = np.mean(
    [j for k in [i['test']['mse'] for i in rkf_knn] for j in k])
rkf_knn_test_mae_mean = np.mean(
    [j for k in [i['test']['mae'] for i in rkf_knn] for j in k])
rkf_knn_test_r2_mean = np.mean(
    [j for k in [i['test']['r2'] for i in rkf_knn] for j in k])

rkf_knn_train_mse_mean = np.mean(
    [j for k in [i['train']['mse'] for i in rkf_knn] for j in k])
rkf_knn_train_mae_mean = np.mean(
    [j for k in [i['train']['mae'] for i in rkf_knn] for j in k])
rkf_knn_train_r2_mean = np.mean(
    [j for k in [i['train']['r2'] for i in rkf_knn] for j in k])

rkf_xgb_test_mse_mean = np.mean(
    [j for k in [i['test']['mse'] for i in rkf_xgb] for j in k])
rkf_xgb_test_mae_mean = np.mean(
    [j for k in [i['test']['mae'] for i in rkf_xgb] for j in k])
rkf_xgb_test_r2_mean = np.mean(
    [j for k in [i['test']['r2'] for i in rkf_xgb] for j in k])

rkf_xgb_train_mse_mean = np.mean(
    [j for k in [i['train']['mse'] for i in rkf_xgb] for j in k])
rkf_xgb_train_mae_mean = np.mean(
    [j for k in [i['train']['mae'] for i in rkf_xgb] for j in k])
rkf_xgb_train_r2_mean = np.mean(
    [j for k in [i['train']['r2'] for i in rkf_xgb] for j in k])

#%%
print('mae',
    rkf_knn_test_mae_mean,
    rkf_xgb_test_mae_mean, '\n',
      'mse',
      rkf_knn_test_mse_mean,
      rkf_xgb_test_mse_mean, '\n'
                             'r2',
      rkf_knn_test_r2_mean,
      rkf_xgb_test_r2_mean, '\n'
      )

#%%
np.sqrt(6.2)

#%%
np.sqrt(mean_mse)


#%%
print(
lr_perf, '\n',
xgb_perf, '\n',
knn_perf_30)

#%%
np.mean([abs(i) for i in
 [-2.01861646, -1.91271436, -2.07313101, -1.92410622, -1.96538945]])

#%%
sparse_perf_dict = {'lr': lr_perf,
                    'knn_30': knn_perf_30,
                    'xgb': xgb_perf}
pd.to_pickle(sparse_perf_dict, 'data/interim/sparse_perf_dict.pickle')

#%%
print('lr',
np.mean(lr_perf['test']['r2']),
np.sqrt(np.mean(lr_perf['test']['mse'])),
np.mean(lr_perf['test']['mae']),
      '\n',
'knn',
np.mean(knn_perf_30['test']['r2']),
np.sqrt(np.mean(knn_perf_30['test']['mse'])),
np.mean(knn_perf_30['test']['mae']),
      '\n',
'xgb',
np.mean(xgb_perf['test']['r2']),
np.sqrt(np.mean(xgb_perf['test']['mse'])),
np.mean(xgb_perf['test']['mae']),
      '\n'
      )
