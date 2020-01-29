import pandas as pd
import numpy as np
import scipy
#%%
merged_df = pd.read_pickle('data/interim/merged_df.pickle')

#%%
missing_df = merged_df[merged_df['RECOMMEND_BBV'].isna()].reset_index(drop=True)
full_df = merged_df[~merged_df['RECOMMEND_BBV'].isna()].reset_index(drop=True)
#%%
remaining = missing_df.isna().sum() / missing_df.shape[0]
remaining

#%%
remaining[remaining<.5].sort_values()

#%%
remaining_row = missing_df.isna().sum(axis=1)/113

#%%
remaining_row.sort_values().plot(kind='hist')
plt.show()

#%%
mask = ~missing_df.isna().iloc[0,:]

missing_df.loc[1:,mask].isna().sum().plot(kind='hist')
plt.show()

#%%
min_missing = []
for i in range(missing_df.shape[0]):
    lmask = ~missing_df.isna().iloc[i, :]
    lmiss = missing_df.loc[:, lmask].isna().sum(axis=1) / sum(lmask)
    lfull = full_df.loc[:, lmask].isna().sum(axis=1) / sum(lmask)
    min_missing.append(
        [i, sum(lmask),
         min(lmiss), min(lfull),
         np.median(lmiss), np.median(lfull),
         sum(lmiss < .5)/len(lmiss), sum(lfull < .5)/len(lfull),
         sum(lmiss < .2)/len(lmiss), sum(lfull < .2)/len(lfull),
         sum(lmiss < .1)/len(lmiss), sum(lfull < .1)/len(lfull)]
    )

#%%
missing_summary = pd.DataFrame(np.array(min_missing),
                               columns=['index', 'n_features',
                                        'min_miss_m', 'min_miss_f',
                                        'med_miss_m', 'med_miss_f',
                                        'perc_5_miss', 'perc_5_full',
                                        'perc_2_miss', 'perc_2_full',
                                        'perc_1_miss', 'perc_1_full'])

#%%
missing_summary = missing_summary.drop(columns=['min_miss_m', 'min_miss_f'])

#%%
missing_summary.sort_values(
    'n_features')[['n_features', 'perc_2_miss']].set_index(
    'n_features'
).plot(
    kind='line')
plt.show()

#%%
missing_summary[missing_summary['perc_2_miss']>.8]['n_features'].max()

#%%
full_df['RECOMMEND_BBV'].plot(kind='hist')
plt.show()

#%%
scipy.special.logit([.001])

#%%
full_df['RECOMMEND_BBV'].min()