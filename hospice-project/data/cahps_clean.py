#%%
import pandas as pd

path = 'data/raw/cms/Hospice_Compare_-_Provider_CAHPS_Hospice_Survey_Data.csv'
cahps_raw = pd.read_csv(path)

# Keep ID, Measure Code, Score only
cahps_mcs = cahps_raw[['CMS Certification Number (CCN)',
                       'Measure Code', 'Score']].copy()

# Reshape wide
cahps_mcs_wide = cahps_mcs.pivot(index='CMS Certification Number (CCN)',
                columns='Measure Code',
                values='Score')

# Recode missing, make numeric
cahps_mcs_wide = cahps_mcs_wide.replace(
    'Not Available', 'nan').replace(
    'Not  Available', 'nan').replace('Not Applicable','nan').astype(float)

# Save to csv
cahps_mcs_wide.to_csv('data/interim/cms/cahps.csv')