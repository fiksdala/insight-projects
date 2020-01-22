#%% Libraries
import pandas as pd
import numpy as np
import datetime as dt
import pickle

#%% Clean PUF
# Read in data
path = 'data/raw/cms/PAC Provider PUF_v20191002.csv'
puf2017raw = pd.read_csv(path)

# Filter out national/state summaries and anything aside from hospice service
# Drop redundant variables
mask = (puf2017raw['Summary Category'] == 'PROVIDER') & \
             (puf2017raw['Service Category'] == 'HOS')

# Drop redundant vars
drop_vars = ['Summary Category', 'State', 'Service Category',
             'Facility Name', 'City', 'ZIP']

puf2017hos = puf2017raw[mask].drop(columns=drop_vars).copy()

# Clean up strings
def replace_func(x):
    x = x.strip()
    x = x.replace('%', '').replace(',', '').replace('*', 'nan')
    x = x.replace('$', '').replace('NA', 'nan')
    x = x.replace('nannan', '')
    return x
puf2017hos = puf2017hos.astype(str).apply(
    lambda x: [replace_func(i) for i in x], axis=0
)

# Make df floats except for id (int)
puf2017hos = puf2017hos.replace('', 'nan').astype(float)
puf2017hos['Provider ID'] = puf2017hos['Provider ID'].astype(str)
puf2017hos.rename(columns={'Provider ID': 'ccn'})

# Pickle to retain dtypes
pickle.dump(puf2017hos, open('data/interim/puf.pickle', 'wb'))

#%% Clean Provider Data
# Import raw data
raw = pd.read_csv('data/raw/cms/Hospice_-_Provider_Data.csv')

#%%
raw['Measure Code'].value_counts()

#%%

# Drop 'DENOMINATOR' variables (raw values will be scaled anyway)
mask = [False if 'DENOMINATOR' in i else True
        for i in raw['Measure Code']]
df_2018 = raw_2018[mask].copy()


#%%
# df_2018['Measure Code'].value_counts()

#%%

# drop extraneous
df_2018.drop(columns=['Facility Name', 'Address Line 2', 'PhoneNumber',
                      'CMS Region', 'Measure Name', 'Footnote',
                      'Start Date', 'End Date'], inplace=True)

# convert to wide format, rename and keep ccn, address, measure code, and score
df_2018.rename(columns={'CMS Certification Number (CCN)': 'cnn'},
               inplace=True)
measures_2018 = ['cnn', 'Measure Code', 'Score']
df_2018_measures = df_2018[measures_2018].pivot(
    index='cnn',
    columns='Measure Code',
    values='Score'
)

ids_2018 = ['cnn', 'Address Line 1', 'City',
            'State', 'Zip Code', 'County Name']
wide_2018 = pd.merge(
    df_2018[ids_2018].drop_duplicates(),
    df_2018_measures,
    on='cnn'
)

#%% checks
wide_2018.dtypes

#%%

# Convert address to latlong
# Drop PR, guam, etc.
drop_st = ['VI', 'GU', 'MP', 'PR']
st_mask = [False if i in drop_st else True for i in wide_2018.State]
wide_2018 = wide_2018[st_mask].copy()

# Use simplemaps dict for fast city lat/long lookup

us_cities = pd.read_csv('data/raw/CMS/uscities.csv')

ll_dict = dict(
    zip(
        [' '.join(i) for i in us_cities[['city', 'state_id']].to_numpy()],
        [(i[0],i[1]) for i in us_cities[['lat', 'lng']].to_numpy()]
    )
)

cities = [' '.join(i) for i in wide_2018[['City', 'State']].to_numpy()]
ll_dict = {k.lower(): v for (k, v) in ll_dict.items()}

wide_2018['lat_long'] = [ll_dict[i.lower()] if i.lower() in ll_dict.keys()
                         else 'None' for i in cities]

# Get remaining city lat/long using county lat/long lookup (less accurate)
# https://en.wikipedia.org/wiki/User:Michael_J/County_table
countylatlong = pd.read_csv('data/raw/CMS/counties.csv',
                         sep=None,
                         engine='python')
countylatlong = countylatlong[['County [2]', 'State',
'Latitude', 'Longitude']]

county_state = [' '.join(i) for i in
                wide_2018[['County Name', 'State']].to_numpy().astype(str)]

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

wide_2018.loc[wide_2018.lat_long=='None','lat_long'] = [
    countyll_dict[i] if i in countyll_dict.keys() else np.nan
    for i in np.array(county_state)[wide_2018.lat_long=='None']]

wide_2018.drop(columns=['Address Line 1', 'City', 'Zip Code'],
               inplace=True)

# Convert 'Not Available' to np.nan
wide_2018.replace('Not Availale', np.nan).dtypes

#%%
wide_2018.dtypes
