import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_validate
import hospice_project.definitions as defs
import hospice_project.data.transformers as t
from hospice_project.data.train_test_split import get_train_test

complete_df = pd.read_pickle('data/interim/complete_df.pickle')
X_raw, y = get_train_test(complete_df,
                      defs.sparse_vars,
                      'would_recommend',
                          return_full=True)

X_raw_er, y_er = get_train_test(complete_df,
                      defs.sparse_vars,
                      'RATING_EST',
                                return_full=True)

steps = [('scaler', t.MyScaler(dont_scale='for_profit')),
             ('knn', t.KNNKeepDf())]
pipe = Pipeline(steps)
pipe.fit(X_raw)
X = pipe.transform(X_raw)
#%%

# RandomSearch best XGBRegressor parameters
params = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.5, 1 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }

rs = RandomizedSearchCV(XGBRegressor(),
                        params)
rs.fit(X, y)

#%%
wr_xgb = XGBRegressor(**rs.best_params_)
wr_xgb.fit(X, y)

pipe.fit(X_raw_er)
X_er = pipe.transform(X_raw_er)

er_xgb = XGBRegressor(**rs.best_params_)
er_xgb.fit(X_er, y_er)


#%%
to_predict = complete_df.loc[complete_df['RECOMMEND_BBV'].isna(),
                             defs.sparse_vars]
pipe.fit(X_raw)
wr_preds = wr_xgb.predict(pipe.transform(to_predict))

pipe.fit(X_raw_er)
er_preds = er_xgb.predict(pipe.transform(to_predict))

#%%
sparse_preds = pd.DataFrame({
    'ccn': complete_df[complete_df['RECOMMEND_BBV'].isna()]['ccn'],
    'would_recommend': wr_preds,
    'RATING_EST': er_preds
})

sparse_preds.to_pickle('models/sparse_preds.pickle')


