import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from hospice_project.data.transformer import MyScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

feature_groups = {
    'Service Quality': ['EMO_REL_BTR', 'RESPECT_BTR', 'SYMPTOMS_BTR',
                        'TEAM_COMM_BTR', 'TIMELY_CARE_BTR', 'TRAINING_BTR'],
    'Services Delivered': ['H_009_01_OBSERVED', 'percSOSDassisLiv',
                           'nurseVisitCtPB', 'socialWorkCtPB',
                           'physicianCtPB'],
    'Services Offered': ['Care_Provided_Assisted_Living_Yes',
                         'Care_Provided_Home_Yes',
                         'Care_Provided_Inpatient_Hospice_Yes',
                         'Care_Provided_Skilled_Nursing_Yes'],
    'Patients Served': ['percBlack', 'percHisp', 'percBen30orFewerDays'],
    'Financial Details': 'totalMedStandPayPB'
}

pca_groups = {
    'Service Quality': ['EMO_REL_BTR', 'RESPECT_BTR', 'SYMPTOMS_BTR',
                        'TEAM_COMM_BTR', 'TIMELY_CARE_BTR', 'TRAINING_BTR'],
    'Services Delivered': ['H_009_01_OBSERVED', 'percSOSDassisLiv',
                           'nurseVisitCtPB', 'socialWorkCtPB',
                           'physicianCtPB']
}

#%%
X_train = pd.read_pickle('data/processed/X_train.pickle')
X_test = pd.read_pickle('data/processed/X_test.pickle')
y_train = pd.read_pickle('data/processed/y_train.pickle')
y_test = pd.read_pickle('data/processed/y_test.pickle')
var_dict = pd.read_pickle('data/interim/var_dict.pickle')
keep_vars = pd.read_pickle('data/interim/r_keep.pickle')

mask_train = ~X_train['RECOMMEND_BBV'].isna()
mask_test = ~X_test['RECOMMEND_BBV'].isna()

#%% make xgb x/y (and make y [0,1]
xgb_x = X_train.loc[mask_train, keep_vars].copy()
xgb_y = y_train[mask_train].to_numpy()/100

#%% Adjust for extreme [0,1] values (see R beta regression document)
xgb_y = (xgb_y * (xgb_y.shape[0] - 1) + 0.5) / xgb_y.shape[0]

#%%
xgb_y

#%% fit transformer
steps = [('scaler', MyScaler(dont_scale=var_dict['dummy_vars'])),
         ('knn', KNNImputer())]
pipe = Pipeline(steps)
pipe.fit(xgb_x)

# PCA
pca = PCA()
pca.fit(pipe.transform(xgb_x)[:,1:7])

#%%
huh = pca.transform(pipe.transform(xgb_x)[:,1:7])

#%%
huh.std(axis=0)


#%%

cov_mat = np.cov(pipe.transform(xgb_x)[:,1:7].T)
#%%
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

#%%
pd.DataFrame(eig_vecs,
             columns=['comp_'+str(i) for i in range(eig_vecs.shape[1])],
             index=xgb_x.columns[1:7])


#%%

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

#%%
gc = GetComponents(pca_groups)

#%%
huhX = pd.DataFrame(pipe.transform(xgb_x),
                    columns=pipe['scaler'].colnames_)
gc.fit(huhX)

#%%
PCs = gc.transform(huhX)

#%%
huhX.to_csv('data/processed/ready_X.csv', index=False)
PCs.to_csv('data/processed/ready_Xpca.csv', index=False)
pd.DataFrame({'y': xgb_y}).to_csv('data/processed/ready_y.csv',
                                  index=False)

