#%%
import pandas as pd
import datetime as dt
from dateutil import relativedelta

#%%

gen_raw = pd.read_csv('data/raw/cms/Hospice_-_General_Information.csv')

# Keep ID, Ownership Type, and Certification Date
own_cert = gen_raw[['CMS Certification Number (CCN)',
                    'Ownership Type', 'Certification Date']].copy()

# Rename ccn
own_cert = own_cert.rename(columns={'CMS Certification Number (CCN)': 'ccn'})

# ccn as string for merging later
own_cert['ccn'] = own_cert['ccn'].astype(str)

#%%
# Make months_operation = months since certification to 1/1/17 (date of
# cahps data start)
cahps_start = dt.datetime.strptime('01/01/2017', '%m/%d/%Y')
own_cert['days_operation'] = [
    [cahps_start - dt.datetime.strptime(i, '%m/%d/%Y')][0].days
    for i in own_cert['Certification Date']
]

#%%
own_cert['ccn'] = [i[1:] if i[0]=='0' else i for i in own_cert['ccn']]

#%%
# pickle subset
pickle.dump(own_cert[['ccn', 'Ownership Type', 'days_operation']],
            open('data/interim/provider_general.pickle', 'wb'))

