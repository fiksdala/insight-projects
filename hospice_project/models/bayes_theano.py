#%% Imports
import pandas as pd
import numpy as np
import pymc3 as pm
from matplotlib import pyplot as plt
from hospice_project.data.transformer import MyScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import seaborn as sns
from theano import shared
from scipy import stats

#%% Data frames and variable lists
X_train = pd.read_pickle('data/processed/X_train.pickle')

# Drop dv and obs missing RECOMMEND_BBV
bayes_x = X_train[r_keep_vars][~X_train['RECOMMEND_BBV'].isna()]

# Assign y var
bayes_y = X_train['RECOMMEND_BBV'][~X_train['RECOMMEND_BBV'].isna()]
bayes_y.reset_index(drop=True, inplace=True)

# Scale and shift, make percentages proportions and shift by .01 to avoid 0s
# This will ensure that we can estimate beta likelihood and posterior
bayes_y = (bayes_y / 100) + .01

# Scale and impute using pipeline
steps = [('scaler', MyScaler(dont_scale=var_dict['dummy_vars'])),
         ('KNNImputer', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(bayes_x)
X = pipe.transform(bayes_x)
X = pd.DataFrame(X, columns=bayes_x.columns)

#%% Build Model

#






