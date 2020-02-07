from sklearn.model_selection import train_test_split
import pandas as pd
import hospice_project.definitions as defs
import hospice_project.data.transformers as t
from sklearn.pipeline import Pipeline



def get_train_test(df, x_vars, y_var,
                   return_train=True,
                   return_test=False,
                   random_state=42,
                   return_full=False):
    """Returns selected dataframe with specified variabes and mask.
    Splits occur before mask and variable selection, so test set remains the
    same across specifications

    dataset will be filtered by complete y_var by default

    if both return_train and return_test:
        returns: X_train, y_train, X_test, y_test
    else:
        returns X, y

    pickel_train/test_csv = saves merged X y to csv as
    '[required_full]_[train/text].csv' in 'processed' folder
    """

    # Get splits
    raw_X_train, raw_X_test, raw_y_train, raw_y_test = train_test_split(
        df,
        df[y_var],
        test_size=.33,
        random_state=random_state,
        stratify=df['State']
    )

    # Apply mask and feature selection
    X_train = raw_X_train.loc[~raw_X_train[y_var].isna(), x_vars]
    X_train.reset_index(drop=True, inplace=True)

    y_train = raw_y_train[~raw_X_train[y_var].isna()]
    y_train.reset_index(drop=True, inplace=True)

    X_test = raw_X_test.loc[~raw_X_test[y_var].isna(), x_vars]
    X_test.reset_index(drop=True, inplace=True)

    y_test = raw_y_test[~raw_X_test[y_var].isna()]
    y_test.reset_index(drop=True, inplace=True)

    if return_train and return_test == False:
        return X_train, y_train

    if return_test and return_train == False:
        return X_test, y_test

    if return_train and return_test:
        return X_train, y_train, X_test, y_test

    if return_full:
        y = df[y_var][~df[y_var].isna()]
        y.reset_index(drop=True, inplace=True)
        return df.loc[~df[y_var].isna(), x_vars], y


def r_out(df, x_vars, y_var, csv_label):


    def to_r(x, y, csv_label):
        """merges y and X and exports csv to data/processed for use in R, etc.
        Use after scaling/imputing"""
        return pd.concat([y, x], axis=1).to_csv(
            'data/processed/' + csv_label + '.csv', index=False)


    X_raw, y = get_train_test(df,
                              x_vars,
                              y_var)

    steps = [('scaler', t.MyScaler(dont_scale=['for_profit'])),
             ('knn', t.KNNKeepDf())]
    pipe = Pipeline(steps)

    pipe.fit(X_raw)
    X = pipe.transform(X_raw)

    to_r(X, y, csv_label)