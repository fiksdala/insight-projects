#%% imports
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pickle

X_train = pd.read_pickle('data/processed/X_train.pickle')
var_dict = pickle.load(open('data/interim/var_dict.pickle', 'rb'))
var_dict['explore']

#%% Explore correlations by category
# Drop vars if corr > .9 for 2+ vars by category

#%% Patient Evaluation Vars
p_evals = ['H_001_01_OBSERVED',
 'H_002_01_OBSERVED',
 'H_003_01_OBSERVED',
 'H_004_01_OBSERVED',
 'H_005_01_OBSERVED',
 'H_006_01_OBSERVED',
 'H_007_01_OBSERVED',
 'H_008_01_OBSERVED',
 'H_009_01_OBSERVED',
 'EMO_REL_BBV',
 'EMO_REL_TBV',
 'RATING_BBV',
 'RATING_TBV',
 'RECOMMEND_BBV',
 'RECOMMEND_TBV',
 'RESPECT_BBV',
 'RESPECT_TBV',
 'SYMPTOMS_BBV',
 'SYMPTOMS_TBV',
 'TEAM_COMM_BBV',
 'TEAM_COMM_TBV',
 'TIMELY_CARE_BBV',
 'TIMELY_CARE_TBV',
 'TRAINING_BBV',
 'TRAINING_TBV'
           ]

# H_ vars do not correlate highly with survey bin vars
plt.figure(figsize=(15,10))
sns.heatmap(X_train[p_evals].astype(float).corr())
plt.show()

# Particularly high (>.85) with ratings and emo
p_evals_cor = X_train[p_evals].astype(float).corr().to_numpy()
np.fill_diagonal(p_evals_cor, 0)
plt.figure(figsize=(15,10))
sns.heatmap(pd.DataFrame(
 np.abs(p_evals_cor) > .85,
 columns=p_evals,
 index=p_evals
).iloc[9:,9:])
plt.show()

# p_evals_keep: drop emo_tbv and ratings
p_evals_keep = [i for i in p_evals
                if i not in ['EMO_REL_TBV', 'RATING_BBV', 'RATING_TBV']]

# Provider Services Offered (all dummies, keep for now)
prov_serv_offered_keep = ['Care_Provided_Assisted_Living_Missing',
 'Care_Provided_Assisted_Living_No',
 'Care_Provided_Assisted_Living_Yes',
 'Care_Provided_Home_Missing',
 'Care_Provided_Home_No',
 'Care_Provided_Home_Yes',
 'Care_Provided_Inpatient_Hospice_Missing',
 'Care_Provided_Inpatient_Hospice_No',
 'Care_Provided_Inpatient_Hospice_Yes',
 'Care_Provided_Inpatient_Hospital_Missing',
 'Care_Provided_Inpatient_Hospital_No',
 'Care_Provided_Inpatient_Hospital_Yes',
 'Care_Provided_Nursing_Facility_Missing',
 'Care_Provided_Nursing_Facility_No',
 'Care_Provided_Nursing_Facility_Yes',
 'Care_Provided_Skilled_Nursing_Missing',
 'Care_Provided_Skilled_Nursing_No',
 'Care_Provided_Skilled_Nursing_Yes',
 'Care_Provided_other_locations_Missing',
 'Care_Provided_other_locations_No',
 'Care_Provided_other_locations_Yes',
                          'Provided_Home_Care_and_other_Missing',
                          'Provided_Home_Care_and_other_No',
                          'Provided_Home_Care_and_other_Yes',
                          'Provided_Home_Care_only_Missing',
                          'Provided_Home_Care_only_No',
                          'Provided_Home_Care_only_Yes'
                          ]

#%% Demographics
# Check for sufficient variation
p_demo = ['percWhite',
 'percBlack',
 'percAsian',
 'percHisp',
 'percNative',
 'percOther']

for i in p_demo:
    X_train[i].plot(kind='hist')
    plt.show()

# Keep White/Black/Hisp for most variation
p_demo_keep = ['percWhite',
 'percBlack',
 'percHisp']

#%% Provider Services Delivered
p_services_delivered = ['percBen7orFewerDays',
 'percBen30orFewerDays',
 'percBen60orFewerDays',
 'percBen180orFewerDays',
'aveNurseMin7dp',
 'aveSocialWork7dp',
 'aveHomeHealth7dp',
 'percSOSDhome',
 'percSOSDassisLiv',
 'percSOSlongtermcare',
 'percSOSskilledNurse',
 'percSOSinpatient',
 'percSOSinpatientHospice',
 'percDeathDisch',
 'percHospLiveDisch',
 'percDaysHospRHC',
'nurseVisitCtPB',
 'socialWorkCtPB',
 'homeHealthCtPB',
 'physicianCtPB'
]

plt.figure(figsize=(15,10))
sns.heatmap(X_train[p_services_delivered].corr())
plt.show()

p_serv_del_corr = X_train[p_services_delivered].corr().to_numpy()
np.fill_diagonal(p_serv_del_corr,0)

plt.figure(figsize=(15,10))
sns.heatmap(p_serv_del_corr)
plt.show()

#%%
# 7/30 + and 60/180 - cor, keep 30 and 180
# Drop discharge and RHC
p_services_delivered_keep = [i for i in p_services_delivered
                             if i not in [
                              'percBen7orFewerDays',
                              'percBen60orFewerDays',
                              'percDeathDisch',
                              'percHospLiveDisch',
                              'percDaysHospRHC'
                             ]]

# Check variation of remaining and drop low variance vars
for var in p_services_delivered_keep:
    print(var)
    X_train[var].plot(kind='hist', title=var)
    plt.show()

#%% drop nurse, percSOSinpatientHospice, percSOSinpatient,
# percSOSskilledNurse, health 7dp
p_services_delivered_keep = [i for i in p_services_delivered_keep
                             if i not in ['aveNurseMin7dp',
                                          'aveHomeHealth7dp',
                                          'percSOSinpatientHospice',
                                          'percSOSinpatient',
                                          'percSOSskilledNurse']]

#%% Finances, keep just totalMedStandPayPB and tot_med_PB
p_finances = ['totalMedStandPayPB',
              'tot_med_PB']

#%% Combine keep to prep for random forest feature selection
keep_vars = p_evals_keep + prov_serv_offered_keep \
            + p_services_delivered_keep + p_finances


#%% Save keep_vars for use later

pickle.dump(keep_vars, open('data/processed/keep_vars.pickle', 'wb'))


