import pandas as pd
import hospice_project.definitions as defs
import hospice_project.data.transformers as t
from hospice_project.data.train_test_split import get_train_test, r_out
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

complete_df = pd.read_pickle('data/interim/complete_df.pickle')
X_raw, y = get_train_test(complete_df,
                      defs.dense_vars,
                      'would_recommend',
                          return_full=True)

X_raw_er, y_er = get_train_test(complete_df,
                      defs.dense_vars,
                      'RATING_EST',
                                return_full=True)

#%%
steps = [('scaler', t.MyScaler(dont_scale='for_profit')),
         ('knn', t.KNNKeepDf())]
pipe_recommend = Pipeline(steps)
pipe_recommend.fit(X_raw)
X_recommend = pipe_recommend.transform(X_raw)

pipe_est_rating = Pipeline(steps)
pipe_est_rating.fit(X_raw_er)
X_est_rating = pipe_est_rating.transform(X_raw)



#%%
lr_recommend = LinearRegression()
lr_est_rating = LinearRegression()

lr_recommend.fit(X_recommend, y)
lr_est_rating.fit(X_est_rating, y_er)

pd.to_pickle(lr_recommend, 'models/lr_recommend.pickle')
pd.to_pickle(lr_est_rating, 'models/lr_est_rating.pickle')
pd.to_pickle(pipe_recommend, 'models/pipe_recommend.pickle')
pd.to_pickle(pipe_est_rating, 'models/pipe_est_rating.pickle')