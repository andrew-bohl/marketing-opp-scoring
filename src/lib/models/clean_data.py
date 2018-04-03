"""  DATA CLEANING MODULE
    Loads and preps data for modeling
"""
import datetime as dt

import pandas as pd
import numpy as np

from src.data import filters
import src.lib.models.utilities as utils


def clean_salesforce_data(client, sql):
    """
    Transforms data for salesforce
    :return: SalesForce dataframe
    """
    def filter_bad_orderids(dataframe, colname, filter_num):
        dataframe['to_keep'] = dataframe[colname].apply(lambda x: len(x) >= filter_num)
        dataframe = dataframe[dataframe['to_keep'] == 1]
        return dataframe

    salesforce_data = utils.load_bigquery_data(client, sql)

    # runs this one first because filters are case sensitive
    salesforce_data = utils.filter_data(filters.NotQualified, salesforce_data)
    salesforce_data = utils.filter_data(filters.LeadType, salesforce_data)

    salesforce_data = utils.standardize_columns(salesforce_data)

    #filters out ids not of len 7
    salesforce_data = filter_bad_orderids(salesforce_data , 'trial_order_id', 7)

    salesforce_data = salesforce_data.sort_values(['trial_order_id', 'lead_createdate'], ascending=False)\
        .groupby('trial_order_id').first().reset_index()

    salesforce_data = utils.convert_cols_to_datetime(salesforce_data)
    salesforce_data['converted_to_opp'] = salesforce_data['converted_to_opp'].apply(lambda x: 1 if x == 'true' else 0)
    salesforce_data['created_date_dt'] = salesforce_data['lead_createdate'].apply(lambda x: x.date())

    def impute_opp_close_date(createdate):
        """calculated a close date"""
        if createdate:
            newdate = createdate + dt.timedelta(days=21)
            return newdate
        else:
            return createdate

    salesforce_data['opp_close_date_impute'] = salesforce_data['created_date_dt'].apply(impute_opp_close_date)
    salesforce_data['opp_close_date_impute'] = pd.to_datetime(salesforce_data['opp_close_date_impute'])
    salesforce_data['opp_close_date_impute'] = salesforce_data['opp_close_date_impute'].\
        combine_first(salesforce_data['opp_closedate'])

    salesforce_data = salesforce_data[salesforce_data['lead_createdate'] <= salesforce_data['opp_close_date_impute']]
    return salesforce_data


def clean_ga_data(client, sql):
    """
    Transforms data for ga session data
    :return: dataframe
    """
    ga_paths = utils.load_bigquery_data(client, sql)
    ga_paths = utils.standardize_columns(ga_paths)
    ga_paths = utils.convert_cols_to_datetime(ga_paths)
    ga_paths['session_date'] = ga_paths['date']

    ga_paths['landingpagepath'] = ga_paths['landingpagepath'].fillna('')
    ga_paths = ga_paths[~ga_paths['landingpagepath'].str.contains('store.volusion.com')]
    ga_paths.to_csv("ga_sessions_20180403.csv")
    return ga_paths


def clean_conversions_data(client, sql):
    """
    Transforms trial conversion data
    :return: dataframe
    """
    trials = utils.load_bigquery_data(client, sql)
    trials = utils.standardize_columns(trials)

    trials = utils.convert_cols_to_datetime(trials)
    return trials


def merge_datasets(salesforce_df, ga_df, conversions_df, cutoff=dt.datetime(2018, 2, 1)):
    """
    :param salesforce: salesforce dataframe
    :param ga: ga sessions dataframe
    :param conversions: warehouse conversions dataframe
    :param cutoff: cut off date of observations to include
    :return: merged dataframe for feature extraction
    """
    #merge sf and conversion object
    merged = salesforce_df.set_index('trial_order_detail_id').join(conversions_df.set_index('trial_id'), how='inner')

    paths_sf = ga_df.set_index('demo_lookup').join(merged, lsuffix='_ga', how='inner')
    paths_sf['date'] = pd.to_datetime(paths_sf['date'])
    paths_sf['session_close_date_diff'] = pd.to_datetime(paths_sf['opp_close_date_impute']) - paths_sf['date']

    session_nums = paths_sf[paths_sf['session_close_date_diff'] >= dt.timedelta(days=3)].reset_index().groupby('index').count()['session_id']

    marketing_sources = ga_df[ga_df['first_demo_session'].notnull()]

    #master dataframe
    data_merged = marketing_sources.set_index('demo_lookup').join(merged, how='inner', lsuffix='_ms')

    final_df = data_merged.join(session_nums, how='left', rsuffix='_count')
    final_df['session_id_count'] = final_df['session_id_count'].fillna(0)
    final_df['session_count'] = final_df['session_id_count']

    final_df['time_mismatch'] = pd.to_datetime(final_df['session_date']) - pd.to_datetime(final_df['trial_date'])
    final_df['time_mismatch'] = final_df['time_mismatch'].apply(lambda x: x.days)
    final_df['trial_date'] = pd.to_datetime(final_df['trial_date'])

    if cutoff:
        final_df = final_df[final_df['trial_date'] < cutoff]

    else:
        # get cut_off date, original cutoff was 2/1/2018
        month_ago = dt.datetime.now().date() - dt.timedelta(days=30)
        final_df = final_df[final_df['trial_date'] < month_ago]

    final_df.loc[final_df['status'] == 'Open', 'converted'] = 0
    final_df = final_df[final_df['converted'] != 0]
    final_df.to_pickle('raw_merged_data')
    return final_df


def create_features(data):

    #transformed_vars
    data['affiliate_touch'] = data['click_id'].notnull()
    data['affiliate_touch'] = data['affiliate_touch'].apply(lambda x: 1 if x else 0)
    t = ['True', 'Yes']
    data['have_products'] = data['have_products__c'].isin(t).apply(lambda x: 1 if x else 0)
    transformed_vars = ['converted', 'affiliate_touch', 'have_products', 'session_count']

    #get dummy variables
    cat_variables = ['landingpagepath', 'devicecategory', 'addistributionnetwork',
                     'medium', 'socialnetwork', 'campaign', 'adkeywordmatchtype',
                     'country_ms', 'lead_type__c', 'leadsource', 'reason_not_qualified__c',
                     'sms_opt_in__c', 'industry__c', 'socialsignup__c', 'gender__c', 'position__c']

    for x in cat_variables:
        data[x] = data[x].astype('category')

    dummies = pd.get_dummies(data[cat_variables])
    joined_dummies = data[transformed_vars].join(dummies)

    features_names = joined_dummies.columns[1:]
    features_set = np.array(joined_dummies[joined_dummies.columns[1:]])
    target_variable = np.array(joined_dummies['converted'])

    np.save('features_names', features_names)
    np.save('raw_features_X', features_set)
    np.save('target_Y', target_variable)

    return features_names, target_variable, features_set


def main():
    pass


if __name__ == '__main__':
    main()
