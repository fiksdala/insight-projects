import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from hospice_project.data.transformer import MyScaler
import pickle
from sklearn.neighbors import NearestNeighbors
import numpy as np

#%% Import data, vars, dicts, etc.
var_dict = pd.read_pickle('data/interim/var_dict.pickle')
r_keep_vars = pickle.load(open('data/interim/r_keep.pickle', 'rb'))
X_train = pd.read_pickle('data/processed/X_train.pickle')
sparse_keep_vars = pd.read_pickle('data/interim/sparse_keep_vars.pickle')

#%% Add model groups
X_train['model_group'] = ['sparse' if i else 'full'
                          for i in X_train['RECOMMEND_BBV'].isna()]

#%% Add >50% missing indicator
X_train['sparse_miss_50'] = X_train[sparse_keep_vars].isna().sum(
    axis=1)/len(sparse_keep_vars) > .5

#%% Get knn_dict for 'full' model_group
full_x = X_train[r_keep_vars][
    ~X_train['RECOMMEND_BBV'].isna()
].reset_index(drop=True)

# Make ccn dict from indices
ccn_dict_full = dict(zip(
    range(full_x.shape[0]),
    X_train[~X_train['RECOMMEND_BBV'].isna()].reset_index(drop=True)['ccn']
))

# set up scale/impute pipeline
steps = [('scaler', MyScaler(dont_scale=var_dict['dummy_vars'])),
         ('KNNImputer', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(full_x)
X = pipe.transform(full_x)
X = pd.DataFrame(X, columns=full_x.columns)

knn_ids = NearestNeighbors(n_neighbors=4)
knn_ids.fit(X)
distances, indices = knn_ids.kneighbors(X)

knn_dict_full = dict(zip(
    [ccn_dict_full[i] for i in range(X.shape[0])],
    [[ccn_dict_full[i] for i in r] for r in indices]
))


#%%
pickle.dump(knn_ids, open('data/interim/knn_ids.pickle', 'wb'))


#%% Get knn_dict for 'sparse' model_group
# impute/scale using full dataset
sparse_impute_df = X_train[sparse_keep_vars].copy()

# set up scale/impute pipeline
steps_sparse = [('scaler', MyScaler(dont_scale=var_dict['dummy_vars'])),
                ('KNNImputer', KNNImputer())]
pipe_sparse = Pipeline(steps_sparse)
pipe_sparse.fit(sparse_impute_df)
X_sparse_vardf = pipe_sparse.transform(sparse_impute_df)
X_sparse_vardf = pd.DataFrame(X_sparse_vardf, columns=sparse_impute_df.columns)

# get knn for sparse_vars from knn fitted to full vars
# Separate sparse and full
# Make sparse ccn dict from indices
X_sparse = X_sparse_vardf.loc[np.array(
    X_train['model_group'] == 'sparse'), :].copy()
X_sparse.reset_index(drop=True, inplace=True)
X_full = X_sparse_vardf.loc[np.array(
    X_train['model_group'] == 'full'), :].copy()
X_full.reset_index(drop=True)

# ccn dicts for sparse and full
ccn_dict_sparse = dict(zip(
    range(X_sparse.shape[0]),
    X_train[X_train['model_group'] == 'sparse'].reset_index(drop=True)['ccn']
))

ccn_dict_not_sparse = dict(zip(
    range(X_full.shape[0]),
    X_train[X_train['model_group'] == 'full'].reset_index(drop=True)['ccn']
))

# Fit the nn on full
knn_sparse = NearestNeighbors(n_neighbors=3)
knn_sparse.fit(X_full)
# Get nn for sparse
distances, indices = knn_sparse.kneighbors(X_sparse)

knn_dict_sparse = dict(zip(
    [ccn_dict_sparse[i] for i in range(X_sparse.shape[0])],
    [[ccn_dict_not_sparse[i] for i in r] for r in indices]
))

pickle.dump(knn_dict_full, open('data/interim/knn_dict_full.pickle', 'wb'))
pickle.dump(knn_dict_sparse, open('data/interim/knn_dict_sparse.pickle', 'wb'))
pickle.dump(knn_sparse, open('data/interim/knn_sparse.pickle', 'wb'))
pickle.dump(knn_ids, open('data/interim/knn_ids.pickle', 'wb'))
pickle.dump(pipe_sparse, open('data/interim/pipe_sparse.pickle', 'wb'))

#%%
pickle.dump(pipe, open('data/interim/pipe.pickle', 'wb'))

#%%#############################################################################
X['ccn'] = [ccn_dict_not_sparse[i] for i in range(X.shape[0])]
X_sparse['ccn'] = [ccn_dict_sparse[i] for i in range(X_sparse.shape[0])]
X_id = X_train[['ccn', 'Facility Name', 'State', 'RECOMMEND_BBV',
                'model_group', 'sparse_miss_50']].copy()

pd.to_pickle(X, 'data/interim/X_full.pickle')
pd.to_pickle(X_sparse, 'data/interim/X_sparse.pickle')
pd.to_pickle(X_id, 'data/interim/X_id.pickle')

#%%



















#%% ols_dfs drop recommend vars and make recommend_bbv Y
knn_x = X_train[r_keep_vars][~X_train['RECOMMEND_BBV'].isna()].reset_index(drop=True)
knn_y = X_train['RECOMMEND_BBV'][~X_train['RECOMMEND_BBV'].isna()].reset_index(drop=True)

#%%

# Make ccn dict from indices
ccn_dict = dict(zip(
    range(knn_x.shape[0]),
    X_train[~X_train['RECOMMEND_BBV'].isna()].reset_index(drop=True)['ccn']
))

#%%
# set up scale/impute pipeline
steps = [('scaler', MyScaler(dont_scale=var_dict['dummy_vars'])),
         ('KNNImputer', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(knn_x)
X = pipe.transform(knn_x)
X = pd.DataFrame(X, columns=knn_x.columns)

#%% knn
knn = KNeighborsClassifier()
knn.fit(X,knn_y)

#%% Get neighbors
knn_ids = NearestNeighbors(n_neighbors=4)
knn_ids.fit(X.iloc[50:,:])
distances, indices = knn_ids.kneighbors(X.iloc[50:,:])

#%% Try with individual
huhd, huhi = knn_ids.kneighbors(X.iloc[:50,:].to_numpy())


#%%
huhi

#%%
k3nn_dict = dict(zip(
    [ccn_dict[i] for i in indices[:, 0]],
    [[ccn_dict[i] for i in r] for r in indices[:, 1:]]
))
k3nn_dict

#%% Combine dfs
id_df = X_train['ccn', 'Facility Name', 'State']

#%%

knn_ids = NearestNeighbors(n_neighbors=4)
knn_ids.fit(X)


