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


#%% Import full dataset (validation uses k-fold with simulated missingness)
complete_df = pd.read_pickle('data/interim/complete_df.pickle')
X_wr, y_wr = get_train_test(complete_df,
                      defs.sparse_vars,
                      'would_recommend',
                          return_full=True)

X_er, y_er = get_train_test(complete_df,
                      defs.sparse_vars,
                      'RATING_EST',
                                return_full=True)

# Define sparse dataset (any missing would_recommend)
sparse_df = complete_df.loc[complete_df['would_recommend'].isna(),
                            defs.sparse_vars]

#%% Define functions to simulate missingness and execute k-fold with
# simulated missingness applied to test fold only

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


def resample_kfold(X, y, sparse_df, model, folds=5, prefix='name_'):
    # Get folds
    folds = ShuffleSplit(n_splits=folds, test_size=(1 / folds))
    folds.get_n_splits(X, y)

    # Set up history/scoring lists
    model_history = []
    train_r2 = []
    train_mse = []
    train_mae = []

    test_r2 = []
    test_mse = []
    test_mae = []
    # execute k-fold
    for train_index, test_index in folds.split(X, y):
        xtrain, xtest = X.iloc[train_index,:], X.iloc[test_index,:]
        ytrain, ytest = y[train_index], y[test_index]

        # Simulate missingness on test fold
        xtest = pd.DataFrame(xtest, columns=X.columns)
        xtest = sim_miss(xtest, sparse_df)

        # Scale/transform xtrain
        steps = [('scaler', t.MyScaler(defs.dummy_vars)),
                 ('knn', t.KNNKeepDf())]
        pipe = Pipeline(steps)
        pipe.fit(xtrain)
        xtrain = pipe.transform(xtrain)

        # scale/impute test (test has simulated missing, imputed on
        # data from training folds)
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

    performance = {prefix+'train_r2': train_r2,
                  prefix+'train_mse': train_mse,
                  prefix+'train_mae': train_mae,
                  prefix+'test_r2': test_r2,
                  prefix+'test_mse': test_mse,
                  prefix+'test_mae': test_mae}
    return performance

#%% Estimate error for would_recommend and RATING_EST
# would_recommend
def get_performance(X, y):
    knn_perf = resample_kfold(X, y, sparse_df, KNeighborsRegressor(
            n_neighbors=30), prefix='knn_')

    # RandomSearch best XGBRegressor parameters
    params = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.5, 1 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

    rs = RandomizedSearchCV(XGBRegressor(),
                            params)
    rs.fit(X.to_numpy(), y)

    xgb_perf = resample_kfold(X, y, sparse_df,
                              XGBRegressor(**rs.best_params_),
                              prefix='xgb_')


    lr_perf = resample_kfold(X, y, sparse_df, LinearRegression(),
                             prefix='lr_')

    # Combine Dicts
    output = lr_perf
    output.update(knn_perf)
    output.update(xgb_perf)

    # Return pandas dataframe
    return pd.DataFrame(output)


#%% get scores
wr_perfs = get_performance(X=X_wr, y=y_wr)
ratings_perfs = get_performance(X=X_er, y=y_er)

#%% Save performance
wr_perfs.to_pickle('models/wr_predict_performance.pickle')
ratings_perfs.to_pickle('models/rate_predict_performance.pickle')


