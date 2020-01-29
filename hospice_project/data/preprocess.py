import pandas as pd
from haversine import haversine_vector, Unit
import numpy as np
import pickle
initial_df = pd.read_pickle('data/interim/merged_df.pickle')
initial_df['ccn'].duplicated().sum()

#Feature Engineering

#%% N facilities in 30, 60, 90 mile radius and min dist
latlongs = initial_df.loc[~initial_df.lat_long.isna(),
    ['ccn', 'lat_long']
]
# latlongs = initial_df['lat_long'][~initial_df.lat_long.isna()]

dist_matrix = np.array([haversine_vector([i]*len(latlongs['lat_long']),
                                         list(latlongs['lat_long']), Unit.MILES)
 for i in list(latlongs['lat_long'])])

# Fill vars
np.fill_diagonal(dist_matrix,10000)
n_win_30 = np.array(dist_matrix < 30).sum(axis=1)
n_win_60 = np.array(dist_matrix < 60).sum(axis=1)
n_win_90 = np.array(dist_matrix < 90).sum(axis=1)

np.fill_diagonal(dist_matrix, np.nan)
nearest = np.nanmin(dist_matrix, axis=1)

# dist_df = pd.DataFrame({
#     'ccn': initial_df['ccn'][~initial_df.lat_long.isna()],
#     'n_win_30': n_win_30,
#     'n_win_60': n_win_60,
#     'n_win_90': n_win_90,
#     'nearest': nearest
# })

latlongs['n_win_30'] = n_win_30
latlongs['n_win_60'] = n_win_60
latlongs['n_win_90'] = n_win_90
latlongs['nearest'] = nearest

#%% Join initial_df and latlongs
initial_df = initial_df.join(latlongs.drop(columns='lat_long').set_index('ccn'),
                             on='ccn')

#%% Charges/distinctBen; totalCharge - totalMedPay/distinctBen; cts/distinctBen
bens = initial_df['distinctBens']
initial_df['totChargePB'] = initial_df['totalCharge']/bens
initial_df['totalMedStandPayPB'] = initial_df['totalMedStandPay']/bens
dpb = (initial_df['totalCharge'] - initial_df['totalMedPay']) / bens
initial_df['tot_med_PB'] = dpb
initial_df['nurseVisitCtPB'] = initial_df['nurseVisitCt']/bens
initial_df['socialWorkCtPB'] = initial_df['socialWorkCt']/bens
initial_df['homeHealthCtPB'] = initial_df['homeHealthCt']/bens
initial_df['physicianCtPB'] = initial_df['physicianCt']/bens


#%% Care_Provided Y/N to Y/N/Miss
cp_vars = [
"Care_Provided_Assisted_Living",
"Care_Provided_Home",
"Care_Provided_Inpatient_Hospice",
"Care_Provided_Inpatient_Hospital",
"Care_Provided_Nursing_Facility",
"Care_Provided_Skilled_Nursing",
"Care_Provided_other_locations",
"Provided_Home_Care_and_other",
"Provided_Home_Care_only"
]
initial_df[cp_vars] = initial_df[cp_vars].fillna('Missing')

#%% Race adjust: make non-white % to deal with large % missing issue
# and fill nan to 0 if total > 95
race_vars = ["percWhite",
"percBlack",
"percAsian",
"percHisp",
"percNative",
"percOther"]

# Fill nan to 0 if total is >95
initial_df.loc[
    initial_df[race_vars].sum(axis=1) > 95,
    race_vars
] = initial_df.loc[
    initial_df[race_vars].sum(axis=1) > 95,
    race_vars
].fillna(0)

initial_df['percNonwhite'] = 100-initial_df['percWhite']

#%% Make H/L Ratio Vars for CAHPS items
tbvs = [i for i in initial_df.columns if 'TBV' in i]
bbvs = [i for i in initial_df.columns if 'BBV' in i]
btr_vars = []
for var in range(len(tbvs)):
    btr = tbvs[var][:-3]+'BTR'
    initial_df[btr] = initial_df[bbvs[var]] / initial_df[tbvs[var]]
    btr_vars.append(btr)

#%% Varlists
cat_vars = [
    'Ownership Type',
    'Care_Provided_Assisted_Living',
    'Care_Provided_Home',
    'Care_Provided_Inpatient_Hospice',
    'Care_Provided_Inpatient_Hospital',
    'Care_Provided_Nursing_Facility',
    'Care_Provided_Skilled_Nursing',
    'Care_Provided_other_locations',
    'Provided_Home_Care_and_other',
    'Provided_Home_Care_only'
]

initial_df = pd.get_dummies(initial_df, columns=cat_vars)

drop_levels = ['Ownership Type_For-Profit',
               'Care_Provided_Assisted_Living_Missing',
               'Care_Provided_Home_Missing',
               'Care_Provided_Inpatient_Hospice_Missing',
               'Care_Provided_Nursing_Facility_Missing',
               'Care_Provided_Skilled_Nursing_Missing',
               'Care_Provided_other_locations_Missing',
               'Provided_Home_Care_and_other_Missing',
               'Provided_Home_Care_only_Missing',
               'Care_Provided_Inpatient_Hospital_Missing'
               ]

reference_level_dict = dict(zip(cat_vars[1:],
                                drop_levels))

dummy_vars = [
    'Ownership Type_Combination Government & Non-Profit',
       'Ownership Type_For-Profit', 'Ownership Type_Government',
       'Ownership Type_Non-Profit', 'Ownership Type_Other',
       'Care_Provided_Assisted_Living_Missing',
       'Care_Provided_Assisted_Living_No', 'Care_Provided_Assisted_Living_Yes',
       'Care_Provided_Home_Missing', 'Care_Provided_Home_No',
       'Care_Provided_Home_Yes', 'Care_Provided_Inpatient_Hospice_Missing',
       'Care_Provided_Inpatient_Hospice_No',
       'Care_Provided_Inpatient_Hospice_Yes',
       'Care_Provided_Inpatient_Hospital_Missing',
       'Care_Provided_Inpatient_Hospital_No',
       'Care_Provided_Inpatient_Hospital_Yes',
       'Care_Provided_Nursing_Facility_Missing',
       'Care_Provided_Nursing_Facility_No',
       'Care_Provided_Nursing_Facility_Yes',
       'Care_Provided_Skilled_Nursing_Missing',
       'Care_Provided_Skilled_Nursing_No', 'Care_Provided_Skilled_Nursing_Yes',
       'Care_Provided_other_locations_Missing',
       'Care_Provided_other_locations_No', 'Care_Provided_other_locations_Yes',
       'Provided_Home_Care_and_other_Missing',
       'Provided_Home_Care_and_other_No', 'Provided_Home_Care_and_other_Yes',
       'Provided_Home_Care_only_Missing', 'Provided_Home_Care_only_No',
       'Provided_Home_Care_only_Yes'
]

# Drop extraneous / duplicate variables:
drop_vars_initial = [
"Address Line 1",
"County Name",
"City",
"Zip Code",
"lat_long",
"totalCharge",
"totalMedStandPay",
"totalMedPay",
"percBenCancer_Prim",
"percBenCOPD_Prim",
"percBenRespFail_Prim",
"percBenDementia_Prim",
"PercBenStroke_Prim",
"PercBenCHF_Prim",
"percBenHypertens_Prim",
"percBenOtherCirc_Prim",
"percBenInfect_Prim",
"percBenMuscSkel_Prim",
"percBenInjury_Prim",
"percBenMotorNeural_Prim",
"percBenDiabetes_Prim",
"percBenBurn_Prim",
"percBenAfterCare_Prim",
"nurseVisitCt",
"socialWorkCt",
"homeHealthCt",
"physicianCt",
"EMO_REL_MBV",
"RATING_MBV",
"RECOMMEND_MBV",
"RESPECT_MBV",
"SYMPTOMS_MBV",
"TEAM_COMM_MBV",
"TIMELY_CARE_MBV",
"TRAINING_MBV"
]

# Prediction variables (including vars that are not 'actionable')
pred_vars_initial = [
'ccn',
'Ownership Type_Combination Government & Non-Profit',
    'Ownership Type_For-Profit',
    'Ownership Type_Government',
    'Ownership Type_Non-Profit',
    'Ownership Type_Other',
"days_operation",
"Facility Name",
"State",
"Average_Daily_Census",
"Care_Provided_Assisted_Living",
"Care_Provided_Home",
"Care_Provided_Inpatient_Hospice",
"Care_Provided_Inpatient_Hospital",
"Care_Provided_Nursing_Facility",
"Care_Provided_Skilled_Nursing",
"Care_Provided_other_locations",
"H_001_01_OBSERVED",
"H_002_01_OBSERVED",
"H_003_01_OBSERVED",
"H_004_01_OBSERVED",
"H_005_01_OBSERVED",
"H_006_01_OBSERVED",
"H_007_01_OBSERVED",
"H_008_01_OBSERVED",
"H_009_01_OBSERVED",
"Pct_Pts_w_Cancer",
"Pct_Pts_w_Circ_Heart_Disease",
"Pct_Pts_w_Dementia",
"Pct_Pts_w_Resp_Disease",
"Pct_Pts_w_Stroke",
"Pct_Pts_w_other_conditions",
"distinctBens",
"epStayCt",
"daysService",
"percBen7orFewerDays",
"percBen30orFewerDays",
"percBen60orFewerDays",
"percBen180orFewerDays",
"percMedAdvBen",
"percDualBen",
"percRuralZipBen",
"aveAge",
"percMale",
"percFemale",
"percWhite",
"percBlack",
"percAsian",
"percHisp",
"percNative",
"percOther",
"aveHCCscore",
"aveNurseMin7dp",
"aveSocialWork7dp",
"aveHomeHealth7dp",
"percSOSDhome",
"percSOSDassisLiv",
"percSOSlongtermcare",
"percSOSskilledNurse",
"percSOSinpatient",
"percSOSinpatientHospice",
"percDeathDisch",
"percHospLiveDisch",
"percDaysHospRHC",
"EMO_REL_BBV",
"EMO_REL_TBV",
"RATING_BBV",
"RATING_TBV",
"RECOMMEND_BBV",
"RECOMMEND_TBV",
"RESPECT_BBV",
"RESPECT_TBV",
"SYMPTOMS_BBV",
"SYMPTOMS_TBV",
"TEAM_COMM_BBV",
"TEAM_COMM_TBV",
"TIMELY_CARE_BBV",
"TIMELY_CARE_TBV",
"TRAINING_BBV",
"TRAINING_TBV",
"n_win_30",
"n_win_60",
"n_win_90",
"nearest",
"totChargePB",
"totalMedStandPayPB",
"tot_med_PB",
"nurseVisitCtPB",
"socialWorkCtPB",
"homeHealthCtPB",
"physicianCtPB"
]

# These are the vars from which I'll build the prediction/exploration tool,
# i.e. 'actionable' vars of interest
explore_vars_initial = [
'ccn',
"Facility Name",
"State",
"Average_Daily_Census",
'Care_Provided_Assisted_Living_Missing',
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
"H_001_01_OBSERVED",
"H_002_01_OBSERVED",
"H_003_01_OBSERVED",
"H_004_01_OBSERVED",
"H_005_01_OBSERVED",
"H_006_01_OBSERVED",
"H_007_01_OBSERVED",
"H_008_01_OBSERVED",
"H_009_01_OBSERVED",
"percBen7orFewerDays",
"percBen30orFewerDays",
"percBen60orFewerDays",
"percBen180orFewerDays",
"aveAge",
"percWhite",
"percBlack",
"percAsian",
"percHisp",
"percNative",
"percOther",
    'percNonwhite',
"aveNurseMin7dp",
"aveSocialWork7dp",
"aveHomeHealth7dp",
"percSOSDhome",
"percSOSDassisLiv",
"percSOSlongtermcare",
"percSOSskilledNurse",
"percSOSinpatient",
"percSOSinpatientHospice",
"percDeathDisch",
"percHospLiveDisch",
"percDaysHospRHC",
"EMO_REL_BBV",
"EMO_REL_TBV",
"RATING_BBV",
"RATING_TBV",
"RECOMMEND_BBV",
"RECOMMEND_TBV",
"RESPECT_BBV",
"RESPECT_TBV",
"SYMPTOMS_BBV",
"SYMPTOMS_TBV",
"TEAM_COMM_BBV",
"TEAM_COMM_TBV",
"TIMELY_CARE_BBV",
"TIMELY_CARE_TBV",
"TRAINING_BBV",
"TRAINING_TBV",
"totChargePB",
"totalMedStandPayPB",
"tot_med_PB",
"nurseVisitCtPB",
"socialWorkCtPB",
"homeHealthCtPB",
"physicianCtPB",
'Provided_Home_Care_and_other_Missing',
       'Provided_Home_Care_and_other_No',
    'Provided_Home_Care_and_other_Yes',
       'Provided_Home_Care_only_Missing',
    'Provided_Home_Care_only_No',
       'Provided_Home_Care_only_Yes',
'EMO_REL_BTR',
 'RATING_BTR',
 'RECOMMEND_BTR',
 'RESPECT_BTR',
 'SYMPTOMS_BTR',
 'TEAM_COMM_BTR',
 'TIMELY_CARE_BTR',
 'TRAINING_BTR'
]

# Make dict of variable information
var_dict = {
    'pred': pred_vars_initial,
    'explore': explore_vars_initial,
    'cat_vars': cat_vars,
    'drop_levels': drop_levels,
    'dummy_vars': dummy_vars,
    'reference_levels': reference_level_dict
}

#%% Drop the initial vars, make float, and save relevant info
# floats make later processing/analysis steps easier
initial_df = initial_df.drop(columns=drop_vars_initial)

not_float = ['ccn', 'Facility Name', 'Address Line 1', 'City',
             'State', 'Zip Code', 'County Name']
floats = [i for i in initial_df.columns if i not in not_float]
initial_df[floats] = initial_df[floats].astype(float)

initial_df.to_pickle('data/interim/initial_df.pickle')
pickle.dump(var_dict, open('data/interim/var_dict.pickle', 'wb'))