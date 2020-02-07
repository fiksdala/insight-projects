#%% Libraries
import pandas as pd
import numpy as np
import datetime as dt
import pickle

#%% Clean Provider Data
# Import raw data
raw = pd.read_csv('data/raw/cms/Hospice_-_Provider_Data.csv')

# Drop 'DENOMINATOR' variables
mask = [False if 'DENOMINATOR' in i else True
        for i in raw['Measure Code']]
provider_all = raw[mask].copy()

#%%

# drop extraneous
provider_all.drop(columns=['Address Line 2', 'PhoneNumber',
                      'CMS Region', 'Measure Name', 'Footnote',
                      'Start Date', 'End Date'], inplace=True)

# convert to wide format, rename and keep ccn, address, measure code, and score
provider_all.rename(columns={'CMS Certification Number (CCN)': 'ccn'},
               inplace=True)
measures = ['ccn', 'Measure Code', 'Score']
provider_measures = provider_all[measures].pivot(
    index='ccn',
    columns='Measure Code',
    values='Score'
)

ids = ['ccn', 'Facility Name', 'Address Line 1', 'City',
            'State', 'Zip Code', 'County Name']
provider_wide = pd.merge(
    provider_all[ids].drop_duplicates(),
    provider_measures,
    on='ccn'
)

#%% checks
provider_wide.dtypes

#%%

# Convert address to latlong
# Drop PR, guam, etc.
drop_st = ['VI', 'GU', 'MP', 'PR']
st_mask = [False if i in drop_st else True for i in provider_wide.State]
provider_wide = provider_wide[st_mask].copy()

# Use simplemaps dict for fast city lat/long lookup

us_cities = pd.read_csv('data/raw/cms/uscities.csv')

ll_dict = dict(
    zip(
        [' '.join(i) for i in us_cities[['city', 'state_id']].to_numpy()],
        [(i[0],i[1]) for i in us_cities[['lat', 'lng']].to_numpy()]
    )
)

cities = [' '.join(i) for i in provider_wide[['City', 'State']].to_numpy()]
ll_dict = {k.lower(): v for (k, v) in ll_dict.items()}

provider_wide['lat_long'] = [ll_dict[i.lower()] if i.lower() in ll_dict.keys()
                         else 'None' for i in cities]

# Get remaining city lat/long using county lat/long lookup (less accurate)
# https://en.wikipedia.org/wiki/User:Michael_J/County_table
countylatlong = pd.read_csv('data/raw/cms/counties.csv',
                         sep=None,
                         engine='python')
countylatlong = countylatlong[['County [2]', 'State',
'Latitude', 'Longitude']]

county_state = [' '.join(i) for i in
                provider_wide[['County Name', 'State']].to_numpy().astype(str)]

countylatlong['Longitude'] = [-float(i[1:-1]) if i[0] != '+' else float(i[1:-1])
 for i in countylatlong.Longitude]

countylatlong['Latitude'] = [-float(i[1:-1]) if i[0] != '+' else float(i[1:-1])
 for i in countylatlong.Latitude]

countyll_dict = dict(
    zip(
        [' '.join(i) for i in
         countylatlong[['County [2]', 'State']].to_numpy().astype(str)],
        [(i[0],i[1]) for i in countylatlong[['Latitude',
                                             'Longitude']].to_numpy()]
    )
)

provider_wide.loc[provider_wide.lat_long=='None','lat_long'] = [
    countyll_dict[i] if i in countyll_dict.keys() else np.nan
    for i in np.array(county_state)[provider_wide.lat_long=='None']]

# Convert 'Not Available' and '*' to np.nan
provider_wide.replace('Not Available', np.nan, inplace=True)
provider_wide.replace('*', np.nan, inplace=True)
provider_wide.replace('Less than 11', np.nan, inplace=True)

#%%
# Define float variables
provider_floats = ['Average_Daily_Census', 'H_001_01_OBSERVED',
                   'H_002_01_OBSERVED', 'H_003_01_OBSERVED',
                   'H_004_01_OBSERVED', 'H_005_01_OBSERVED',
                   'H_006_01_OBSERVED', 'H_007_01_OBSERVED',
                   'H_008_01_OBSERVED', 'H_009_01_OBSERVED',
                   'Pct_Pts_w_Cancer', 'Pct_Pts_w_Circ_Heart_Disease',
                   'Pct_Pts_w_Dementia', 'Pct_Pts_w_Resp_Disease',
                   'Pct_Pts_w_Stroke'
                   ]

provider_wide[provider_floats] = provider_wide[provider_floats].astype(float)

#%%
# drop leading 0s in ccn
provider_wide['ccn'] = [i[1:] if i[0]=='0' else i
                        for i in provider_wide.ccn.astype(str)]

#%%
# Pickle pre-processed
pickle.dump(provider_wide, open('data/interim/provider.pickle', 'wb'))
