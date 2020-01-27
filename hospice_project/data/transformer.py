#%% Imports
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

#%%
# Must scale continuous only
class MyScaler(BaseEstimator, TransformerMixin):
    def __init__(self,
                 dummy_vars=None):
        self.dummy_vars = dummy_vars
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        if self.dummy_vars is not None:
            self.scaler.fit(X[[i for i in X.columns
                               if i not in self.dummy_vars]])
            self.colnames_ = [i for i in X.columns
                              if i not in self.dummy_vars] + [
                i for i in X.columns if i in self.dummy_vars
            ]
        else:
            self.scaler.fit(X)
            self.colnames_ = X.columns
        return self

    def transform(self, X, y=None, **fit_params):
        if self.dummy_vars is not None:
            X_scaled = self.scaler.transform(
                X[[i for i in X.columns if i not in self.dummy_vars]]
            )
            output = np.concatenate([X_scaled,
                                     X[[i for i in X.columns
                                        if i in self.dummy_vars]]
                                     ], axis=1)
        else:
            output = self.scaler.transform(X)

        return pd.DataFrame(output,
                            columns=self.colnames_)