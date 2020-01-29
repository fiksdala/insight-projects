import pandas as pd
from sklearn.model_selection import train_test_split

initial_df = pd.read_pickle('data/interim/initial_df.pickle')

#%%
initial_df.index.duplicated().sum()
#%% Set aside final test set (model tuning validated via kfold on X_train)
X_train, X_test, y_train, y_test = train_test_split(
    initial_df,
    initial_df['RECOMMEND_BBV'],
    test_size=.33,
    random_state=42,
    stratify=initial_df['State']
)

X_train.to_pickle('data/processed/X_train.pickle')
X_test.to_pickle('data/processed/X_test.pickle')
y_train.to_pickle('data/processed/y_train.pickle')
y_test.to_pickle('data/processed/y_test.pickle')

# export X_train to csv for use in R
X_train.to_csv('data/processed/X_train.csv', index=False)