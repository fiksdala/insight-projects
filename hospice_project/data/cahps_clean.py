#%%
import pandas as pd

path = 'data/raw/cms/Hospice_Compare_-_Provider_CAHPS_Hospice_Survey_Data.csv'
cahps_raw = pd.read_csv(path)

# Keep ID (and rename cnn), Measure Code, Score only
cahps_mcs = cahps_raw[['CMS Certification Number (CCN)',
                       'Measure Code', 'Score']].copy()
cahps_mcs.rename(columns={'CMS Certification Number (CCN)': 'ccn'},
                 inplace=True)

# Reshape wide
cahps_mcs_wide = cahps_mcs.pivot(index='ccn',
                columns='Measure Code',
                values='Score')

#%%
cahps_mcs_wide.reset_index(inplace=True)

#%%
# ccn to str
cahps_mcs_wide['ccn'] = cahps_mcs_wide['ccn'].astype(str)

#%%
# Final adjustments
mask = cahps_mcs_wide=='Not Available'
cahps_mcs_wide[mask] = np.nan
mask = cahps_mcs_wide=='Not  Available'
cahps_mcs_wide[mask] = np.nan
mask = cahps_mcs_wide=='Not Applicable'
cahps_mcs_wide[mask] = np.nan

#%%
# Convert numeric and drop empty vars
cahps_mcs_wide.drop(columns='EMO_REL_MBV')

cahps_mcs_wide.iloc[:, 1:] = cahps_mcs_wide.iloc[:, 1:].astype(float)
cahps_mcs_wide['ccn'] = cahps_mcs_wide['ccn'].astype(str)

#%%
# pickle
pickle.dump(cahps_mcs_wide, open('data/interim/cahps.pickle', 'wb'))
