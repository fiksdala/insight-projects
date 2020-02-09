#%% Imports
import pandas as pd
import numpy as np
import hospice_project.definitions as defs
import hospice_project.data.transformers as t
from hospice_project.data.train_test_split import get_train_test, r_out
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

complete_df = pd.read_pickle('data/interim/complete_df.pickle')
X_raw, y = get_train_test(complete_df,
                      defs.dense_vars,
                      'would_recommend',
                          return_full=True)
r_out(complete_df, defs.dense_vars, 'would_recommend', 'would_recommend')

X_raw_er, y_er = get_train_test(complete_df,
                      defs.dense_vars,
                      'RATING_EST',
                                return_full=True)
r_out(complete_df, defs.dense_vars, 'RATING_EST', 'RATING_EST')

#%% would_recommend
def lr_insight_wr():
    """Return 5-fold cross validation scores r2, mae, rmse"""
    steps = [('scaler', t.MyScaler(dont_scale='for_profit')),
             ('knn', t.KNNKeepDf())]
    pipe = Pipeline(steps)
    pipe.fit(X_raw)
    X = pipe.transform(X_raw)

    lr = LinearRegression()
    lr.fit(X, y)
    cv_results = cross_validate(lr, X, y,
                            scoring=['r2', 'neg_mean_squared_error',
                                     'neg_mean_absolute_error'],
                            return_train_score=True)
    output = pd.DataFrame(
        {'train_r2': [cv_results['train_r2'].mean()],
         'train_rmse': [np.mean(
             [np.sqrt(abs(i))
              for i in cv_results['train_neg_mean_squared_error']])],
         'train_mae': [abs(cv_results['train_neg_mean_absolute_error'].mean())],
         'test_r2': [cv_results['test_r2'].mean()],
         'test_rmse': [np.mean(
             [np.sqrt(abs(i))
              for i in cv_results['test_neg_mean_squared_error']])],
         'test_mae': [abs(cv_results['test_neg_mean_absolute_error'].mean())]
         },
        index=['LR']
    )
    return output


def lgbm_insight_wr():
    """Return 5-fold cross validation scores r2, mae, rmse"""
    steps = [('scaler', t.MyScaler(dont_scale='for_profit')),
             ('knn', t.KNNKeepDf())]
    pipe = Pipeline(steps)
    pipe.fit(X_raw)
    X = pipe.transform(X_raw)

    # Run once to get ideal parameters
    # params = {
    #     'max_bin': [10, 20, 50, 100, 255],
    #     'num_leaves': [5, 10, 31, 50],
    #     'bagging_fraction': [.1, .3, .5, .7, 1]
    # }

    # lgb_q = LGBMRegressor(objective='quantile')
    #
    # gs = RandomizedSearchCV(lgb_q, params,
    #                         scoring=['r2', 'neg_mean_squared_error',
    #                                  'neg_mean_absolute_error'],
    #                         refit='neg_mean_squared_error'
    #                         )
    # gs.fit(X.to_numpy(), y)
    lgbm = LGBMRegressor(num_leaves=50,
                         max_bin=100,
                         bagging_fraction=0.1,
                         objective='quantile')

    cv_results = cross_validate(lgbm, X.to_numpy(), y,
                                scoring=['r2', 'neg_mean_squared_error',
                                         'neg_mean_absolute_error'],
                                return_train_score=True)

    output = pd.DataFrame(
        {'train_r2': [cv_results['train_r2'].mean()],
         'train_rmse': [np.mean(
             [np.sqrt(abs(i))
              for i in cv_results['train_neg_mean_squared_error']])],
         'train_mae': [abs(cv_results['train_neg_mean_absolute_error'].mean())],
         'test_r2': [cv_results['test_r2'].mean()],
         'test_rmse': [np.mean(
             [np.sqrt(abs(i))
              for i in cv_results['test_neg_mean_squared_error']])],
         'test_mae': [abs(cv_results['test_neg_mean_absolute_error'].mean())]
         },
        index=['LGBM']
    )
    return output

#%% est_rating

def lr_insight_er():
    """Return 5-fold cross validation scores r2, mae, rmse"""
    steps = [('scaler', t.MyScaler(dont_scale='for_profit')),
             ('knn', t.KNNKeepDf())]
    pipe = Pipeline(steps)
    pipe.fit(X_raw_er)
    X = pipe.transform(X_raw_er)

    lr = LinearRegression()
    lr.fit(X, y_er)
    cv_results = cross_validate(lr, X, y_er,
                            scoring=['r2', 'neg_mean_squared_error',
                                     'neg_mean_absolute_error'],
                            return_train_score=True)
    output = pd.DataFrame(
        {'train_r2': [cv_results['train_r2'].mean()],
         'train_rmse': [np.mean(
             [np.sqrt(abs(i))
              for i in cv_results['train_neg_mean_squared_error']])],
         'train_mae': [abs(cv_results['train_neg_mean_absolute_error'].mean())],
         'test_r2': [cv_results['test_r2'].mean()],
         'test_rmse': [np.mean(
             [np.sqrt(abs(i))
              for i in cv_results['test_neg_mean_squared_error']])],
         'test_mae': [abs(cv_results['test_neg_mean_absolute_error'].mean())]
         },
        index=['LR']
    )
    return output


def lgbm_insight_er():
    """Return 5-fold cross validation scores r2, mae, rmse"""
    steps = [('scaler', t.MyScaler(dont_scale='for_profit')),
             ('knn', t.KNNKeepDf())]
    pipe = Pipeline(steps)
    pipe.fit(X_raw_er)
    X = pipe.transform(X_raw_er)

    # Run once to get ideal parameters
    # params = {
    #     'max_bin': [10, 20, 50, 100, 255],
    #     'num_leaves': [5, 10, 31, 50],
    #     'min_data_in_leaf': [10, 20, 30],
    #     'bagging_fraction': [.1, .3, .5, .7, 1]
    # }

    # lgb_q = LGBMRegressor(objective='quantile')

    # gs = RandomizedSearchCV(lgb_q, params,
    #                         scoring=['r2', 'neg_mean_squared_error',
    #                                  'neg_mean_absolute_error'],
    #                         refit='neg_mean_squared_error'
    #                         )
    # gs.fit(X, y_er)

    lgbm = LGBMRegressor(num_leaves=50,
                         max_bin=100,
                         bagging_fraction=0.1,
                         objective='quantile')

    cv_results = cross_validate(lgbm, X.to_numpy(), y_er,
                                scoring=['r2', 'neg_mean_squared_error',
                                         'neg_mean_absolute_error'],
                                return_train_score=True)

    output = pd.DataFrame(
        {'train_r2': [cv_results['train_r2'].mean()],
         'train_rmse': [np.mean(
             [np.sqrt(abs(i))
              for i in cv_results['train_neg_mean_squared_error']])],
         'train_mae': [abs(cv_results['train_neg_mean_absolute_error'].mean())],
         'test_r2': [cv_results['test_r2'].mean()],
         'test_rmse': [np.mean(
             [np.sqrt(abs(i))
              for i in cv_results['test_neg_mean_squared_error']])],
         'test_mae': [abs(cv_results['test_neg_mean_absolute_error'].mean())]
         },
        index=['LGBM']
    )
    return output

#%% get estimates
would_recommend_perf = pd.concat([
    lr_insight_wr(),
    lgbm_insight_wr()],
    axis=0
)

rating_est_perf = pd.concat([
    lr_insight_er(),
    lgbm_insight_er()],
    axis=0
)

#%%
print(would_recommend_perf, '\n',
      rating_est_perf)

#%%
insight_compare = {'would_recommend': would_recommend_perf,
                   'rating_est': rating_est_perf}
pd.to_pickle(insight_compare, 'models/insight_compare.pickle')



