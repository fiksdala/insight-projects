import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

initial_df = pd.read_pickle('data/interim/initial_df.pickle')
keep_vars = pd.read_pickle('data/processed/keep_vars.pickle')
sparse_keep_vars = pd.read_pickle('data/interim/sparse_keep_vars.pickle')

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

#%%
sim_miss_df_full = initial_df[sparse_keep_vars].copy()
complete_df = sim_miss_df_full[~initial_df['RECOMMEND_BBV'].isna()].copy()
sparse_df = sim_miss_df_full[initial_df['RECOMMEND_BBV'].isna()].copy()

#%%
def sim_miss(complete_df, sparse_df):
    """Simulate missing data. Assumes MCAR which may not be valid, so use
    caution with inference"""
    # get missing percentages of each column
    missing_perc = sparse_df.isna().sum()/sparse_df.shape[0]

    # apply missingness percentages of each columns randomly
    for mp in missing_perc.index:
        vals = np.random.uniform(0, 1, size=complete_df.shape[0])
        mask = vals < missing_perc[mp]
        complete_df.loc[mask, mp] = np.nan

    # return complete df with simulated missingness
    return complete_df