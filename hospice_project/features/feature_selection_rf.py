#%%
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from hospice_project.data.transformer import MyScaler

keep_vars = pickle.load(open('data/processed/keep_vars.pickle', 'rb'))
X_train = pd.read_pickle('data/processed/X_train.pickle')[keep_vars]

# Drop obs missing RECOMMEND_BBV
X_train = X_train[~X_train['RECOMMEND_BBV'].isna()]
#%%
rf_x = X_train[[i for i in keep_vars
                 if i not in ['RECOMMEND_BBV', 'RECOMMEND_TBV']]]
rf_y = X_train['RECOMMEND_BBV']

steps = [('scaler', MyScaler()),
         ('KNNImputer', KNNImputer())]
pipeline = Pipeline(steps)

#%%
# Bootstrap 100 samples, record feature importance, and average score
pipeline.fit(rf_x)
rf = RandomForestClassifier(max_depth=7)
rf.fit(pipeline.transform(rf_x), rf_y)

#%%
feature_importances = []
for i in range(100):


#%%
loopx = rf_x.sample(5)
print(rf_y[loopx.index])

#%%
huh = rf_x.sample(5)

#%%
initial_df = pd.read_pickle('data/interim/initial_df.pickle')
pd.Series(initial_df.index).duplicated().sum()

#%%
initial_df.duplicated().sum()
