#%% libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

#%% Read in data and dictionaries
pre_drop = pickle.load(open('data/interim/pre_drop.pickle', 'rb'))
cahps = pickle.load(open('data/interim/cahps.pickle', 'rb'))
provider = pickle.load(open('data/interim/provider.pickle', 'rb'))
provider_general = pickle.load(
    open('data/interim/provider_general.pickle', 'rb')
)
puf = pickle.load(open('data/interim/puf.pickle', 'rb'))
cahps_dict = pickle.load(open('data/interim/cahps_dict.pickle', 'rb'))
provider_dict = pickle.load(open('data/interim/provider_dict.pickle', 'rb'))


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

#%%
# merged_df['county'] = merged_df['County Name'] + ' ' + merged_df['State']

#%%
pre_drop.pop(-3)
merged_df.drop(columns=pre_drop, inplace=True)
merged_df.drop(columns='County Name')


#%% Make dfs for neg_rec
neg_rec = merged_df[~merged_df.RECOMMEND_BBV.isna()]

#%%
pd.Series(neg_rec.isna().sum()[neg_rec.isna().sum()/neg_rec.shape[0]<.2]).index[40:]

#%%



#%%
# Initial keep variables:
initial_keep = ['ccn', 'State', 'Ownership Type', 'days_operation',
       'Average_Daily_Census', 'H_001_01_OBSERVED',
       'H_002_01_OBSERVED', 'H_003_01_OBSERVED', 'H_004_01_OBSERVED',
       'H_005_01_OBSERVED', 'H_006_01_OBSERVED', 'H_007_01_OBSERVED',
       'H_008_01_OBSERVED', 'H_009_01_OBSERVED', 'totalCharge',
 'totalMedStandPay', 'totalMedPay', 'percRuralZipBen', 'aveAge', 'percMale',
 'aveHomeHealth7dp', 'percSOSDhome', 'percSOSDassisLiv', 'percSOSlongtermcare',
       'percSOSskilledNurse', 'percSOSinpatient', 'percSOSinpatientHospice',
'EMO_REL_BBV',
       'EMO_REL_TBV', 'RECOMMEND_BBV', 'RESPECT_BBV',
       'RESPECT_MBV', 'RESPECT_TBV', 'SYMPTOMS_BBV', 'SYMPTOMS_MBV',
       'SYMPTOMS_TBV', 'TEAM_COMM_BBV', 'TEAM_COMM_MBV', 'TEAM_COMM_TBV',
       'TIMELY_CARE_BBV', 'TIMELY_CARE_MBV', 'TIMELY_CARE_TBV', 'TRAINING_BBV',
       'TRAINING_MBV', 'TRAINING_TBV', 'distinctBens', 'epStayCt'
 ]

#%%
prelim_df = neg_rec[initial_keep].copy()

#%%
prelim_df.reset_index(drop=True, inplace=True)

#%%
# Make total ben scaled payment var (totalMedStandPay/distinctBens)
prelim_df['medStanPayPerBen'] = prelim_df['totalMedStandPay']/prelim_df['distinctBens']


# Make X and Y for train_test_split
X = prelim_df.drop(columns=['totalCharge', 'totalMedStandPay', 'totalMedPay',
                            'RECOMMEND_BBV']).copy()
Y = prelim_df[~prelim_df['State'].isna()]['RECOMMEND_BBV'].copy()

# Drop missing State
X = X[~X['State'].isna()]


#%%
# Prep for initial simple HLM model: train/test/split, simple mean impute,
# scale

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y, test_size=0.33, random_state=42)


#%%
X_train.iloc[:,:3]

#%%
# Come back and define the pipeline formally later
# for now, just split into cats and conts
to_scale = [i for i in X_train.columns if i not in
            ['ccn', 'State', 'Ownership Type']]
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
scaler = StandardScaler()
X_train = X_train.copy()
X_train[to_scale] = imp_mean.fit_transform(X_train[to_scale])
X_train[to_scale] = scaler.fit_transform(X_train[to_scale])

#%%
X_train['neg_rec'] = Y
X_train.to_csv('data/processed/mvp.csv', index=False)

#%%
X_train['Ownership Type'].value_counts()

#%%

