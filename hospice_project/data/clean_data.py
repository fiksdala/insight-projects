#%% Import
from hospice_project import definitions as defs
import pandas as pd
import numpy as np
import datetime as dt

#%% CAHPS Cleanup
def clean_cahps():
    """Returns cahps data in wide format that is ready to merge with other
    CMS data sources"""
    # read in data
    cahps_raw = pd.read_csv(defs.cahps_path)

    # Keep ID (and rename cnn), Measure Code, Score only
    cahps_mcs = cahps_raw[['CMS Certification Number (CCN)',
                           'Measure Code', 'Score']].copy()
    cahps_mcs.rename(columns={'CMS Certification Number (CCN)': 'ccn'},
                     inplace=True)

    # Reshape wide
    cahps_mcs_wide = cahps_mcs.pivot(index='ccn',
                                     columns='Measure Code',
                                     values='Score')

    cahps_mcs_wide.reset_index(inplace=True)

    # ccn to str (will help merging later)
    cahps_mcs_wide['ccn'] = cahps_mcs_wide['ccn'].astype(str)

    # Final adjustments, convert 'not available' etc. to missing
    mask = cahps_mcs_wide == 'Not Available'
    cahps_mcs_wide[mask] = np.nan
    mask = cahps_mcs_wide == 'Not  Available'
    cahps_mcs_wide[mask] = np.nan
    mask = cahps_mcs_wide == 'Not Applicable'
    cahps_mcs_wide[mask] = np.nan

    # Convert numeric and drop empty vars
    cahps_mcs_wide.drop(columns='EMO_REL_MBV')

    cahps_mcs_wide.iloc[:, 1:] = cahps_mcs_wide.iloc[:, 1:].astype(float)
    cahps_mcs_wide['ccn'] = cahps_mcs_wide['ccn'].astype(str)

    # pickle
    # pickle.dump(cahps_mcs_wide, open('data/interim/cahps.pickle', 'wb'))

    return cahps_mcs_wide

#%% Provider Data Cleanup
def clean_provider():
    """Returns provider dataset prepped for merging"""
    # Import raw data
    raw = pd.read_csv(defs.provider_path)

    # Drop 'DENOMINATOR' variables
    mask = [False if 'DENOMINATOR' in i else True
            for i in raw['Measure Code']]
    provider_all = raw[mask].copy()

    # drop extraneous
    provider_all.drop(columns=['Address Line 2', 'PhoneNumber',
                               'CMS Region', 'Measure Name', 'Footnote',
                               'Start Date', 'End Date'], inplace=True)

    # convert to wide format, rename and keep ccn, address, measure code, and score
    provider_all.rename(columns={'CMS Certification Number (CCN)': 'ccn'},
                        inplace=True)
    measures = ['ccn', 'Measure Code', 'Score']
    provider_measures = provider_all[measures].pivot(
        index='ccn',
        columns='Measure Code',
        values='Score'
    )

    ids = ['ccn', 'Facility Name', 'Address Line 1', 'City',
           'State', 'Zip Code', 'County Name']
    provider_wide = pd.merge(
        provider_all[ids].drop_duplicates(),
        provider_measures,
        on='ccn'
    )

    # Convert address to latlong
    # Drop PR, guam, etc.
    drop_st = ['VI', 'GU', 'MP', 'PR']
    st_mask = [False if i in drop_st else True for i in provider_wide.State]
    provider_wide = provider_wide[st_mask].copy()

    # Use simplemaps dict for fast city lat/long lookup
    us_cities = pd.read_csv(defs.simple_maps_path)

    # Make lat/long dict
    ll_dict = dict(
        zip(
            [' '.join(i) for i in us_cities[['city', 'state_id']].to_numpy()],
            [(i[0], i[1]) for i in us_cities[['lat', 'lng']].to_numpy()]
        )
    )

    # prep cities for mapping
    cities = [' '.join(i) for i in provider_wide[['City', 'State']].to_numpy()]
    ll_dict = {k.lower(): v for (k, v) in ll_dict.items()}

    # get latlongs from cities
    provider_wide['lat_long'] = [ll_dict[i.lower()] if i.lower() in ll_dict.keys()
                                 else 'None' for i in cities]

    # Get remaining city lat/long using county lat/long lookup (less accurate)
    # https://en.wikipedia.org/wiki/User:Michael_J/County_table
    countylatlong = pd.read_csv(defs.county_ll_path,
                                sep=None,
                                engine='python')
    # Relevant variables
    countylatlong = countylatlong[['County [2]', 'State',
                                   'Latitude', 'Longitude']]
    # Join county/state for mapping
    county_state = [' '.join(i) for i in
                    provider_wide[['County Name', 'State']].to_numpy().astype(str)]

    # Get lat/long and make numeric from countylatlong
    countylatlong['Longitude'] = [-float(i[1:-1]) if i[0] != '+' else float(i[1:-1])
                                  for i in countylatlong.Longitude]

    countylatlong['Latitude'] = [-float(i[1:-1]) if i[0] != '+' else float(i[1:-1])
                                 for i in countylatlong.Latitude]

    # Make dict of county lat/longs for mapping
    countyll_dict = dict(
        zip(
            [' '.join(i) for i in
             countylatlong[['County [2]', 'State']].to_numpy().astype(str)],
            [(i[0], i[1]) for i in countylatlong[['Latitude',
                                                  'Longitude']].to_numpy()]
        )
    )

    # Replace missing with county lat/longs or 'none' if missing county
    provider_wide.loc[provider_wide.lat_long == 'None', 'lat_long'] = [
        countyll_dict[i] if i in countyll_dict.keys() else np.nan
        for i in np.array(county_state)[provider_wide.lat_long == 'None']]

    # Convert 'Not Available' and '*' to np.nan
    provider_wide.replace('Not Available', np.nan, inplace=True)
    provider_wide.replace('*', np.nan, inplace=True)
    provider_wide.replace('Less than 11', np.nan, inplace=True)

    # Define float variables
    provider_floats = ['Average_Daily_Census', 'H_001_01_OBSERVED',
                       'H_002_01_OBSERVED', 'H_003_01_OBSERVED',
                       'H_004_01_OBSERVED', 'H_005_01_OBSERVED',
                       'H_006_01_OBSERVED', 'H_007_01_OBSERVED',
                       'H_008_01_OBSERVED', 'H_009_01_OBSERVED',
                       'Pct_Pts_w_Cancer', 'Pct_Pts_w_Circ_Heart_Disease',
                       'Pct_Pts_w_Dementia', 'Pct_Pts_w_Resp_Disease',
                       'Pct_Pts_w_Stroke'
                       ]

    provider_wide[provider_floats] = provider_wide[provider_floats].astype(float)

    # drop leading 0s in ccn
    provider_wide['ccn'] = [i[1:] if i[0] == '0' else i
                            for i in provider_wide.ccn.astype(str)]

    # Pickle pre-processed
    # pickle.dump(provider_wide, open('data/interim/provider.pickle', 'wb'))

    return provider_wide

#%% Facility Data Cleanup
def clean_facility():
    """Returns facility data prepped for merging (ccn, ownership type,
    months of operation only)"""
    gen_raw = pd.read_csv(defs.facility_data_path)

    # Keep ID, Ownership Type, and Certification Date
    own_cert = gen_raw[['CMS Certification Number (CCN)',
                        'Ownership Type', 'Certification Date']].copy()

    # Rename ccn
    own_cert = own_cert.rename(columns={'CMS Certification Number (CCN)': 'ccn'})

    # ccn as string for merging later
    own_cert['ccn'] = own_cert['ccn'].astype(str)

    # Make months_operation = months since certification to 1/1/17 (date of
    # cahps data start)
    cahps_start = dt.datetime.strptime('01/01/2017', '%m/%d/%Y')
    own_cert['days_operation'] = [
        [cahps_start - dt.datetime.strptime(i, '%m/%d/%Y')][0].days
        for i in own_cert['Certification Date']
    ]

    own_cert['ccn'] = [i[1:] if i[0] == '0' else i for i in own_cert['ccn']]

    # pickle subset
    # pickle.dump(own_cert[['ccn', 'Ownership Type', 'days_operation']],
    #             open('data/interim/provider_general.pickle', 'wb'))

    return own_cert[['ccn', 'Ownership Type', 'days_operation']]


#%% Hospice Public Use Cleanup
def clean_puf():
    """Cleans hospice public use file and preps for merging"""

    # Read in data
    puf2017raw = pd.read_csv(defs.puf_path)

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
    puf2017hos = puf2017hos.rename(columns={'Provider ID': 'ccn'})
    puf2017hos['ccn'] = [str(i).split('.')[0] for i in puf2017hos.ccn]

    # Drop vars with 100% missing
    puf2017hos = puf2017hos.loc[:, puf2017hos.isna().sum() / puf2017hos.shape[0] < 1]

    # Rename long vars
    puf_dict = {v: k for k, v in defs.puf_dict.items()}
    puf2017hos = puf2017hos.rename(columns=puf_dict)

    return puf2017hos

#%% Merge
def merge_cms():
    """Merges CMS hospice files and merges/drops duplicate rows.
    Also makes variable summary df to summarize missingness etc."""

    # Read in data and dictionaries
    cahps = clean_cahps()
    provider = clean_provider()
    provider_general = clean_facility()
    puf = clean_puf()
    cahps_dict = defs.cahps_dict
    provider_dict = defs.provider_dict
    puf_dict = defs.provider_dict

    # Merge all and drop pre_drop
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

    # Deal with duplicate ccn's that have different values
    def collapse_rows(x):
        x = x.dropna()
        try:
            x.upper()
            return x[0]
        except:
            return min(x, default=np.nan)

    to_merge = merged_df[merged_df['ccn'].duplicated(keep=False)]
    collapsed_rows = []
    for i in to_merge['ccn'].unique():
        collapsed_rows.append(
            to_merge[to_merge['ccn'] == i].apply(collapse_rows, axis=0)
        )

    merged_rows = pd.DataFrame(np.array(collapsed_rows),
                               columns=to_merge.columns)
    merged_df = pd.concat([merged_df[~merged_df['ccn'].duplicated(keep=False)],
                           merged_rows], axis=0)

    # Observations must have name and location
    merged_df = merged_df[~merged_df['State'].isna()]

    # Reset index from merges and cuts
    merged_df.reset_index(drop=True, inplace=True)

    # Make data source variable
    df_var = ['cahps' if i in cahps.columns
              else 'provider' if i in provider.columns
    else 'provider_general' if i in provider_general.columns
    else 'puf' for i in merged_df.columns[1:]]

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
        merged_df[i].isna().sum() / merged_df.shape[0]
        for i in merged_df.columns[1:]]

    # Sentiment Vars
    sentiments = ['RECOMMEND_BBV', 'RECOMMEND_MBV', 'RECOMMEND_TBV',
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

    missing_df = pd.DataFrame(np.transpose(s_missing),
                              columns=sentiments)

    var_missingness = pd.concat([var_summary,
                                 missing_df], axis=1)

    # Export to csv to explore manually
    var_missingness.to_csv('data/interim/var_missingness.csv',
                           index=False)

    return merged_df

# Merge dfs and pickle to interim
merged_df = merge_cms()
merged_df.to_pickle('data/interim/merged_df.pickle')
