import pandas as pd
import pickle
from matplotlib import pyplot as plt

#%%
initial_df = pd.read_pickle('data/interim/initial_df.pickle')

#%%
missing_df = initial_df[initial_df['RECOMMEND_BBV'].isna()].reset_index(drop=True)
full_df = initial_df[~initial_df['RECOMMEND_BBV'].isna()].reset_index(drop=True)

#%%
miss_sum = pd.DataFrame({
    'sparse': missing_df.isna().sum(axis=0)/missing_df.shape[0],
    'full': full_df.isna().sum(axis=0)/full_df.shape[0]
}).sort_values('sparse')

#%%
for i in miss_sum[miss_sum['sparse']<.4].index:
    print(i)

#%% Cut

cut_nomiss = [i for i in miss_sum.index if (('_No' in i) | ('_Missing' in i))
 & ('Ownership' not in i)]

cut_desciptive = ['ccn', 'State', 'Facility Name']
all_cuts = cut_nomiss + cut_desciptive
sparse_keep = [i for i in miss_sum[miss_sum['sparse']<.4].index
               if i not in all_cuts]

#%%
pickle.dump(sparse_keep, open('data/interim/sparse_keep_vars.pickle',
                              'wb'))

#%%
plt.hist(missing_df.isna().sum(axis=1)/missing_df.shape[1])
plt.show()