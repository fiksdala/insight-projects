#%% Imports
import pandas as pd
import pickle
from hospice_project.data.transformer import MyScaler

#%% Read in merged_df
merged_df = pd.read_pickle('data/interim/merged_df.pickle')

#%% Varselect and transform
# Display df: Facility name ccn state

disp_df = merged_df[['ccn', 'State', 'Facility Name', 'County Name',
                     'RECOMMEND_BBV']]
disp_df.to_pickle('data/processed/disp_df.pickle')

#%%
disp_df[disp_df['State']=='AL']

#%%
interim_df = pd.read_pickle('data/interim/initial_df.pickle')

#%%
merged_df[merged_df['Facility Name'].duplicated()][['Facility Name', 'State']]