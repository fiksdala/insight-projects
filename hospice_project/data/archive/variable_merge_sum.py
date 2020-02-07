#%% libraries
import pandas as pd
import numpy as np
import pickle

#%% Read in data and dictionaries
cahps = pickle.load(open('data/interim/cahps.pickle', 'rb'))
provider = pickle.load(open('data/interim/provider.pickle', 'rb'))
provider_general = pickle.load(
    open('data/interim/provider_general.pickle', 'rb')
)
puf = pickle.load(open('data/interim/puf.pickle', 'rb'))
cahps_dict = pickle.load(open('data/interim/cahps_dict.pickle', 'rb'))
provider_dict = pickle.load(open('data/interim/provider_dict.pickle', 'rb'))
puf_dict = pickle.load(open('data/interim/puf_dict.pickle', 'rb'))

#%% # Merge all and drop pre_drop
merged_df = provider_general.join(
    provider.set_index('ccn'),
    on='ccn'
).join(
    puf.set_index('ccn'),
    on='ccn'
).join(
    cahps.set_index('ccn'),
    on='ccn'
)
merged_df.drop_duplicates(inplace=True)

def collapse_rows(x):
    x = x.dropna()
    try:
        x.upper()
        return x[0]
    except:
        return min(x, default=np.nan)

#%%
to_merge = merged_df[merged_df['ccn'].duplicated(keep=False)]
collapsed_rows = []
for i in to_merge['ccn'].unique():
    collapsed_rows.append(
        to_merge[to_merge['ccn']==i].apply(collapse_rows, axis=0)
    )

#%%
merged_rows = pd.DataFrame(np.array(collapsed_rows),
                           columns=to_merge.columns)
merged_df = pd.concat([merged_df[~merged_df['ccn'].duplicated(keep=False)],
                       merged_rows], axis=0)


#%%
# Observations must have name and location
merged_df = merged_df[~merged_df['State'].isna()]
# Reset index from merges and cuts
merged_df.reset_index(drop=True, inplace=True)
merged_df.to_pickle('data/interim/merged_df.pickle')

#%%
# df by var list
df_var = ['cahps' if i in cahps.columns
 else 'provider' if i in provider.columns
 else 'provider_general' if i in provider_general.columns
 else 'puf'
 for i in merged_df.columns[1:]]

#%%
# Make var summary df
var_summary = pd.DataFrame({
    'var': merged_df.columns[1:],
    'df': df_var})

# Add dict_values
var_summary['desc'] = [
    provider_dict[i] if i in provider_dict.keys()
    else cahps_dict[i] if i in cahps_dict.keys()
    else puf_dict[i] if i in puf_dict.keys()
    else '' for i in var_summary['var']
                       ]

# perc_missing_all
var_summary['TOTAL'] = [
    merged_df[i].isna().sum()/merged_df.shape[0]
    for i in merged_df.columns[1:]]


#%%
# Sentiment Vars
sentiments  = ['RECOMMEND_BBV', 'RECOMMEND_MBV', 'RECOMMEND_TBV',
 'RATING_BBV', 'RATING_MBV', 'RATING_TBV']


s_missing = np.array(
    [
        [
            merged_df[
                ~merged_df[s].isna()
            ][v].isna().sum() / \
            merged_df[~merged_df[s].isna()].shape[0]
            for v in var_summary['var']
        ]
        for s in sentiments
    ]
)

#%%
missing_df = pd.DataFrame(np.transpose(s_missing),
                          columns=sentiments)

#%%
var_missingness = pd.concat([var_summary,
           missing_df], axis=1)

var_missingness.to_csv('data/interim/var_missingness.csv',
                       index=False)







