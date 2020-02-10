#%% Import
import hospice_project.definitions as defs
import hospice_project.data.transformers as t
import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit

merged_df = pd.read_pickle('data/interim/merged_df.pickle')


#%% Distance-based features from lat/long
def add_dist_features(df):
    """Adds distance features based on lat/long to df:
    n_win_30/60/90: Number of facilities within 30/60/90 miles
    nearest: distance in miles to nearest facility (note latlongs based on
    city/county so this is not exact and has some noise)"""

    # N facilities in 30, 60, 90 mile radius and min dist
    latlongs = df.loc[~df.lat_long.isna(),
                      ['ccn', 'lat_long']
    ]

    # Get distance matrix in miles for all facilities
    dist_matrix = np.array([haversine_vector([i] * len(latlongs['lat_long']),
                                             list(latlongs['lat_long']),
                                             Unit.MILES)
                            for i in list(latlongs['lat_long'])])

    # Fill diagonal with huge number, then get n within m distance
    np.fill_diagonal(dist_matrix, 10000)
    # within 30/60/90 miles
    n_win_30 = np.array(dist_matrix < 30).sum(axis=1)
    n_win_60 = np.array(dist_matrix < 60).sum(axis=1)
    n_win_90 = np.array(dist_matrix < 90).sum(axis=1)
    # Fill diagonal with missing, then get nearest facility distance
    np.fill_diagonal(dist_matrix, np.nan)
    nearest = np.nanmin(dist_matrix, axis=1)

    # Assign to latlong df
    latlongs['n_win_30'] = n_win_30
    latlongs['n_win_60'] = n_win_60
    latlongs['n_win_90'] = n_win_90
    latlongs['nearest'] = nearest

    # Join df and latlongs
    df = df.join(latlongs.drop(columns='lat_long').set_index('ccn'),
                 on='ccn')
    return df


#%% Per-beneficiary features
def add_per_ben(df):
    """Converts count-level data to per-beneficiary measures. Returns
    full df with original and added features"""

    # Distinct beneficiaries
    bens = df['distinctBens']

    # Charges per beneficieary
    df['totChargePB'] = df['totalCharge'] / bens
    df['totalMedStandPayPB'] = df['totalMedStandPay'] / bens

    # Total - medicare per beneficiary
    df['tot_med_PB'] = (df['totalCharge'] - df['totalMedPay']) / bens

    # Service counts per beneficiary
    df['nurseVisitCtPB'] = df['nurseVisitCt'] / bens
    df['socialWorkCtPB'] = df['socialWorkCt'] / bens
    df['homeHealthCtPB'] = df['homeHealthCt'] / bens
    df['physicianCtPB'] = df['physicianCt'] / bens

    return df


#%% Add missing category to services provided
def add_missing_cat_services_provided(df):
    """Replaces nan with 'Missing' for 'services provided' variables."""

    # Care_Provided Y/N to Y/N/Miss
    cp_vars = [
        "Care_Provided_Assisted_Living",
        "Care_Provided_Home",
        "Care_Provided_Inpatient_Hospice",
        "Care_Provided_Inpatient_Hospital",
        "Care_Provided_Nursing_Facility",
        "Care_Provided_Skilled_Nursing",
        "Care_Provided_other_locations",
        "Provided_Home_Care_and_other",
        "Provided_Home_Care_only"
    ]
    df[cp_vars] = df[cp_vars].fillna('Missing')
    return df


#%% Custom demographic imputation
def impute_demo(df):
    """Imputes demographic variables to reasonable estimates. CMC suppresses
    some values to protect privacy. If categories with values account for
    >= 95% of possibilities, impute remaining nans with 0. This reduces
    nans without relying on imputation, which may have issues due to
    the peculiar suppression patterns"""

    race_vars = ["percWhite",
                 "percBlack",
                 "percAsian",
                 "percHisp",
                 "percNative",
                 "percOther"]

    # Fill nan to 0 if total is >95
    df.loc[
        df[race_vars].sum(axis=1) > 95,
        race_vars
    ] = df.loc[
        df[race_vars].sum(axis=1) > 95,
        race_vars
    ].fillna(0)

    # Make 'nonwhite' variable
    df['percNonwhite'] = 100 - df['percWhite']

    return df


#%% Make Low-to-High ratio vars for service quality measures
def add_quality_ratios(df):
    """
    NOTE: Deprecated in favor of quality_adjsuted

    Adds quality ratio vars, equal to 1-MinMaxScale(BB/TB)
    This gives a single indicator of the spread between low and high bins
    in a single measure and helps avoid collinearity later on"""

    mm = t.MyMinMax(defs.ratio_vars)
    # Make H/L Ratio Vars for CAHPS items
    tbvs = [i for i in df.columns if 'TBV' in i]
    bbvs = [i for i in df.columns if 'BBV' in i]
    for var in range(len(tbvs)):
        btr = tbvs[var][:-3] + 'BTR'
        df[btr] = df[bbvs[var]] / df[tbvs[var]]

    mm.fit(df)

    return mm.transform(df)


def quality_adjusted(df):
    """Service quality top box scores adjusted for bottom box scores:
    [service]_adjusted = service_tbv/1+.01*service_bbv
    Note: replaces add_quality_ratios"""

    # Get top/bottom box var names
    tbvs = [i for i in df.columns if 'TBV' in i]
    bbvs = [i for i in df.columns if 'BBV' in i]

    # Create adjusted scores
    for var in range(len(tbvs)):
        adj = tbvs[var][:-3] + 'ADJ'
        df[adj] = df[tbvs[var]] / (1 + .01*df[bbvs[var]])

    return df


def rating_estimate(df):
    """Returns df with RATING_EST (sum of weighted median bin scores)"""
    low = df['RATING_BBV'] * .01 * 3.5
    mid = df['RATING_MBV'] * .01 * 7.5
    high = df['RATING_TBV'] * .01 * 9.5

    df['RATING_EST'] = low+mid+high
    return df


def usually_always(df):
    """Returns df with managed_[sentiment] vars which is the sum
    of 'usually' and 'always' scores"""
    df['MANAGED_EMO_REL'] = df['EMO_REL_TBV']  # No middle bin for this var
    df['MANAGED_RESPECT'] = df['RESPECT_TBV'] + df['RESPECT_MBV']
    df['MANAGED_SYMPTOMS'] = df['SYMPTOMS_TBV'] + df['SYMPTOMS_MBV']
    df['MANAGED_TEAM_COMM'] = df['TEAM_COMM_TBV'] + df['TEAM_COMM_MBV']
    df['MANAGED_TIMELY_CARE'] = df['TIMELY_CARE_TBV'] + df['TIMELY_CARE_MBV']
    df['MANAGED_TRAINING'] = df['TRAINING_TBV'] + df['TRAINING_MBV']
    return df


def add_recommend(df):
    """adds general 'would recommend' variable (probably+definitly)"""
    df['would_recommend'] = df['RECOMMEND_MBV'] + df['RECOMMEND_TBV']
    return df


def for_profit_dummy(df):
    """Adds dummy variable for 'for_profit' vs not"""
    df['for_profit'] = [1 if i=='For-Profit' else 0
                        for i in df['Ownership Type']]
    return df

def add_dummies(df):
    """adds dummies for cat vars"""
    cat_vars = [
        'Ownership Type',
        'Care_Provided_Assisted_Living',
        'Care_Provided_Home',
        'Care_Provided_Inpatient_Hospice',
        'Care_Provided_Inpatient_Hospital',
        'Care_Provided_Nursing_Facility',
        'Care_Provided_Skilled_Nursing',
        'Care_Provided_other_locations',
        'Provided_Home_Care_and_other',
        'Provided_Home_Care_only'
    ]

    return pd.get_dummies(df, columns=cat_vars)



#%% Put them all together and save full feature df
feature_functions = [add_dist_features, add_per_ben,
                     add_missing_cat_services_provided, impute_demo,
                     quality_adjusted, rating_estimate,
                     add_quality_ratios, usually_always,
                     for_profit_dummy, add_recommend,
                     add_dummies]

def add_all_features(df, feature_function_list):
    """Applies feature functions sequentially"""

    output = df
    for ff in feature_function_list:
        output = ff(output)

    return output

complete_df = add_all_features(merged_df, feature_functions)
complete_df.to_pickle('data/interim/complete_df.pickle')
complete_df.to_pickle('models/complete_df.pickle')




