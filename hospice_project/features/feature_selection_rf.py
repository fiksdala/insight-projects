#%% Imports
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from hospice_project.data.transformer import MyScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

var_dict = pd.read_pickle('data/interim/var_dict.pickle')

#%% Read in X_train with vars selected in feature_selection_manual.py
keep_vars = pickle.load(open('data/processed/keep_vars.pickle', 'rb'))
X_train = pd.read_pickle('data/processed/X_train.pickle')
X_train_keepvars = X_train[keep_vars+['State']]

# Drop obs missing the DV RECOMMEND_BBV
X_train_keepvars = X_train_keepvars[~X_train_keepvars['RECOMMEND_BBV'].isna()]
X_train_keepvars.reset_index(drop=True, inplace=True)

#%% Make train/validation
X_fs_train, X_fs_val, y_fs_train, y_fs_val = train_test_split(
    X_train_keepvars,
    X_train_keepvars['RECOMMEND_BBV'],
    test_size=.33,
    random_state=42,
    stratify=X_train_keepvars['State']
)

#%% export train/val to csv for use in R

# X_fs_train
r_steps = [('scaler', MyScaler(dummy_vars=var_dict['dummy_vars'])),
           ('KNNImputer', KNNImputer())]
r_pipe = Pipeline(r_steps)
r_pipe.fit(X_fs_train.drop(columns='State'))
to_R_train = pd.DataFrame(r_pipe.transform(X_fs_train.drop(columns='State')),
                    columns=r_pipe['scaler'].colnames_)
r_drop = [i for i in to_R_train.columns if i in var_dict['drop_levels']]
to_R_train.drop(columns=r_drop+['RECOMMEND_TBV'],
          inplace=True)
to_R_train['State'] = X_fs_train['State'].to_numpy()
to_R_train.to_csv('data/processed/to_R_train.csv', index=False)

# X_fs_val
r_pipe = Pipeline(r_steps)
r_pipe.fit(X_fs_val.drop(columns='State'))
to_R_val = pd.DataFrame(r_pipe.transform(X_fs_val.drop(columns='State')),
                    columns=r_pipe['scaler'].colnames_)
r_drop = [i for i in to_R_val.columns if i in var_dict['drop_levels']]
to_R_val.drop(columns=r_drop+['RECOMMEND_TBV'],
          inplace=True)
to_R_val['State'] = X_fs_val['State'].to_numpy()
to_R_val.to_csv('data/processed/to_R_val.csv', index=False)

# Full X_Train
r_pipe = Pipeline(r_steps)
r_pipe.fit(X_train_keepvars.drop(columns='State'))
to_R = pd.DataFrame(r_pipe.transform(X_train_keepvars.drop(columns='State')),
                    columns=r_pipe['scaler'].colnames_)
r_drop = [i for i in to_R.columns if i in var_dict['drop_levels']]
to_R.drop(columns=r_drop+['RECOMMEND_TBV'],
          inplace=True)
to_R['State'] = X_train_keepvars['State'].to_numpy()
to_R.to_csv('data/processed/to_R.csv', index=False)

#%% R keep vars
r_keep_vars = ["H_001_01_OBSERVED",
"H_002_01_OBSERVED",
"H_003_01_OBSERVED",
"H_004_01_OBSERVED",
"H_005_01_OBSERVED",
"H_006_01_OBSERVED",
"H_007_01_OBSERVED",
"H_008_01_OBSERVED",
"H_009_01_OBSERVED",
"EMO_REL_BTR",
"RESPECT_BTR",
"SYMPTOMS_BTR",
"TEAM_COMM_BTR",
"TIMELY_CARE_BTR",
"TRAINING_BTR",
"percWhite",
"percBlack",
"percHisp",
"percBen30orFewerDays",
"percBen180orFewerDays",
"aveSocialWork7dp",
"percSOSDhome",
"percSOSDassisLiv",
"percSOSlongtermcare",
"nurseVisitCtPB",
"socialWorkCtPB",
"homeHealthCtPB",
"physicianCtPB",
"totalMedStandPayPB",
"tot_med_PB",
"Care_Provided_Assisted_Living_Yes",
"Care_Provided_Home_Yes",
"Care_Provided_Inpatient_Hospice_Yes",
"Care_Provided_Inpatient_Hospital_Yes",
"Care_Provided_Nursing_Facility_Yes",
"Care_Provided_Skilled_Nursing_Yes",
"Care_Provided_other_locations_Yes",
"Provided_Home_Care_and_other_Yes",
"Provided_Home_Care_only_Yes"]

#%%
# Loop DF: drop recommend vars and make recommend_bbv Y
rf_x = X_fs_train[r_keep_vars]
rf_y = X_fs_train['RECOMMEND_BBV']

# set up scale/impute pipeline
steps = [('scaler', MyScaler(dummy_vars=var_dict['dummy_vars'])),
         ('KNNImputer', KNNImputer())]
pipeline = Pipeline(steps)

#%%
# Test run with very rough hyperparameter specification
# pipeline.fit(rf_x)
# # max_depth of 7 prevents overfitting
# rf = RandomForestClassifier(max_depth=7)
# rf.fit(pipeline.transform(rf_x), rf_y)

#%% Bootstrap feature_importances: 100 models of 1000 samples each (with
# replacement)
feature_importances = []
for i in range(100):
    i_x = rf_x.sample(1000)
    i_y = rf_y[i_x.index]
    i_pipe = Pipeline(steps)
    i_pipe.fit(i_x)
    i_rf = RandomForestClassifier(max_depth=7)
    i_rf.fit(i_pipe.transform(i_x), i_y)
    feature_importances.append(i_rf.feature_importances_)

#%% Explore/Visualize feature_importance
# feature importance df
fi_df = pd.DataFrame(np.array(feature_importances),
             columns=rf_x.columns)

# feature importance mean/sd
fi_df_msd = pd.concat([fi_df.mean(),
                       fi_df.std()], axis=1)
fi_df_msd.columns = ['mean', 'sd']
fi_df_msd = fi_df_msd.sort_values('mean', ascending=False)
fi_df_msd.plot(kind='line', yerr='sd')
plt.show()

#%%
r_keep_vars = ["H_009_01_OBSERVED",
"EMO_REL_BTR",
"RESPECT_BTR",
"SYMPTOMS_BTR",
"TEAM_COMM_BTR",
"TIMELY_CARE_BTR",
"TRAINING_BTR",
"percBlack",
"percHisp",
"percBen30orFewerDays",
"percSOSDassisLiv",
"nurseVisitCtPB",
"socialWorkCtPB",
"physicianCtPB",
"totalMedStandPayPB",
"Care_Provided_Assisted_Living_Yes",
"Care_Provided_Home_Yes",
"Care_Provided_Inpatient_Hospice_Yes",
"Care_Provided_Skilled_Nursing_Yes"]

pickle.dump(r_keep_vars, open('data/interim/r_keep.pickle', 'wb'))