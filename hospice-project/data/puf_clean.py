#%% Libraries
import pandas as pd
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