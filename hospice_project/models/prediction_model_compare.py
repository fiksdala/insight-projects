import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_validate
import hospice_project.definitions as defs
import hospice_project.data.transformers as t
from hospice_project.data.train_test_split import get_train_test
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import ShuffleSplit
import copy


#%%
complete_df = pd.read_pickle('data/interim/complete_df.pickle')
X_raw, y = get_train_test(complete_df,
                      defs.sparse_vars,
                      'would_recommend')

X_raw_er, y_er = get_train_test(complete_df,
                      defs.sparse_vars,
                      'RATING_EST')

sparse_df = complete_df[~complete_df['would_recommend'].isna()]

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
        steps = [('scaler', t.MyScaler(defs.var_dict['dummy_vars'])),
                 ('knn', t.KNNKeepDf())]
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
knn_mhist_30, knn_perf_30 = resample_kfold(complete_df, y_complete,
                                   sparse_df, KNeighborsRegressor(
        n_neighbors=30
    ))

# RandomSearch best XGBRegressor parameters
params = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.5, 1 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

rs = RandomizedSearchCV(XGBRegressor(),
                        params)
rs.fit(complete_df, y_complete)

xgb_mhist, xgb_perf = resample_kfold(complete_df, y_complete,
                                     sparse_df,
                                     XGBRegressor(**rs.best_params_))


lr_mhist, lr_perf = resample_kfold(complete_df, y_complete,
                                   sparse_df, LinearRegression())


#%%
def get_wr_scores():
    steps = [('scaler', t.MyScaler(dont_scale='for_profit')),
             ('knn', t.KNNKeepDf())]
    pipe = Pipeline(steps)
    pipe.fit(X_raw)
    X = pipe.transform(X_raw)

    lr_mhist, lr_perf = resample_kfold(X, y,
                                       sparse_df, LinearRegression())

    output = pd.DataFrame(lr_perf['test'], index='lr')
    return output

get_wr_scores()


#%%
