#%% Imports
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

#%%
# Must scale continuous only
class MyScaler(BaseEstimator, TransformerMixin):
    """Subsets based on 'subset', then standard scaler, but ignores specified
    vars and recombines with scaled vars.
    Returns df with scaled values and original values specified in dont_scale.
    Inverse transform returns inverse transformed df AFTER subset"""

    def __init__(self, dont_scale=None):
        self.dont_scale = dont_scale
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        if self.dont_scale is not None:
            self.scaler.fit(X[[i for i in X.columns
                               if i not in self.dont_scale]])
            self.colnames_ = [i for i in X.columns
                              if i not in self.dont_scale] + [
                i for i in X.columns if i in self.dont_scale
            ]
        else:
            self.scaler.fit(X)
            self.colnames_ = X.columns
        return self

    def transform(self, X, y=None, **fit_params):
        if self.dont_scale is not None:
            X_scaled = self.scaler.transform(
                X[[i for i in X.columns if i not in self.dont_scale]]
            )
            output = np.concatenate([X_scaled,
                                     X[[i for i in X.columns
                                        if i in self.dont_scale]]
                                     ], axis=1)
        else:
            output = self.scaler.transform(X)

        return pd.DataFrame(output,
                            columns=self.colnames_)

    def inverse_transform(self, X, y=None):
        if self.dont_scale is not None:
            X_inverse = self.scaler.inverse_transform(
                X[[i for i in X.columns if i not in self.dont_scale]]
            )
            output = np.concatenate([X_inverse,
                                     X[[i for i in X.columns
                                        if i in self.dont_scale]]
                                     ], axis=1)
        else:
            output = self.scaler.inverse_transform(X)
        return output

class KNNKeepDf(BaseEstimator, TransformerMixin):
    """KNN imputer, but returns DF and retains column names"""
    def __init__(self):
        self.colnames_ = []
        self.knn = KNNImputer()

    def fit(self, X, y=None):
        self.colnames_ = X.columns
        self.knn.fit(X)
        return self

    def transform(self, X, y=None, **fit_params):
        output = pd.DataFrame(self.knn.transform(X),
                              columns=self.colnames_)
        return output

class GetComponents(BaseEstimator, TransformerMixin):
    """Replaces feature groups with standardized first PCA component"""
    def __init__(self, feature_group_dict, drop_original_vars=True):
        self.feature_group_dict = feature_group_dict
        self.drop_original_vars = drop_original_vars
        self.pca_objects_ = {}
        self.pca_scalers_ = {}

    def fit(self, X, y=None):
        for group in self.feature_group_dict.keys():
            groupX = X[self.feature_group_dict[group]]
            pca = PCA(n_components=1)
            scaler = StandardScaler()
            pca.fit(groupX)
            self.pca_objects_[group] = pca
            self.pca_scalers_[group] = scaler.fit(pca.transform(groupX))
        return self

    def transform(self, X, y=None):
        for group in self.pca_objects_:
            groupX = X[self.feature_group_dict[group]]
            pc = self.pca_objects_[group].transform(groupX)
            X[group] = self.pca_scalers_[group].transform(pc)
        if self.drop_original_vars:
            for var in self.feature_group_dict.keys():
                X.drop(columns=self.feature_group_dict[var],
                       inplace=True)
        return X

class MyMinMax(BaseEstimator, TransformerMixin):
    """Returns df with transformed 1-(0-1 min max scaled) variables specified"""
    def __init__(self, min_max_vars):
        self.min_max_vars = min_max_vars
        self.mm = MinMaxScaler()

    def fit(self, X, y=None):
        self.mm.fit(X[self.min_max_vars])
        return self

    def transform(self, X, y=None):
        X[self.min_max_vars] = 1 - self.mm.transform(X[self.min_max_vars])
        return X

def get_xy(df, y, mask, vars):
    """Returns specified X and y with reset indices."""
    X = df.loc[mask, vars].reset_index(drop=True)
    y = df.loc[mask, y].reset_index(drop=True)
    return X, y