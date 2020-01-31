#%% Imports
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

#%%
# Must scale continuous only
class MyScaler(BaseEstimator, TransformerMixin):
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
