"""  DATA CLEANING MODULE
    Loads and preps data for modeling
"""
import datetime as dt
import logging as log

import numpy as np
from sklearn.decomposition import IncrementalPCA
import pandas as pd
import urllib.parse as url

from src.data import filters
import src.lib.models.utilities as utils


def clean_salesforce_data(client, sql, output_path):
    """Transforms data for salesforce

    :return: SalesForce dataframe
    """
    def filter_bad_orderids(dataframe, colname, filter_num):
        """filter out bad orderids"""
        dataframe['to_keep'] = dataframe[colname].apply(lambda x: len(x) >= filter_num)
        dataframe = dataframe[dataframe['to_keep'] == 1]
        return dataframe

    salesforce_data = utils.load_bigquery_data(client, sql)

    # runs this one first because filters are case sensitive
    salesforce_data = utils.filter_data(filters.NotQualified, salesforce_data)
    salesforce_data = utils.filter_data(filters.LeadType, salesforce_data)
    salesforce_data = utils.standardize_columns(salesforce_data)

    #filters out ids not of len 7

    salesforce_data = filter_bad_orderids(salesforce_data, 'trial_order_id', 7)
    salesforce_data = salesforce_data.sort_values(['trial_order_id', 'lead_createdate'], ascending=False)\
        .groupby('trial_order_id').first().reset_index()

    salesforce_data = utils.convert_cols_to_datetime(salesforce_data)

    #create hour of day feature
    salesforce_data["hour_of_day"] = salesforce_data["lead_createdate"].apply(lambda x: x.hour)

    print(salesforce_data['converted_to_opp'].sum())
    print("Number of converted in sample")

    salesforce_data['created_date_dt'] = salesforce_data['lead_createdate'].apply(lambda x: x.date())

    def impute_opp_close_date(createdate):
        """calculated a close date"""
        if createdate:
            newdate = createdate + dt.timedelta(days=14)
            return newdate
        else:
            return createdate

    salesforce_data['lead_close_date_impute'] = salesforce_data['lead_createdate'].apply(impute_opp_close_date)
    salesforce_data['lead_close_date_impute'] = pd.to_datetime(salesforce_data['lead_close_date_impute'])
    salesforce_data['lead_close_date_impute'] = salesforce_data['opp_createdate'].\
        combine_first(salesforce_data['lead_close_date_impute'])

    salesforce_data['lead_close_date_impute'] = salesforce_data['lead_close_date_impute'].apply(lambda x: x.date())
    salesforce_data['lead_createdate'] = pd.to_datetime(salesforce_data['lead_createdate'])
    salesforce_data['lead_createdate'] = salesforce_data['lead_createdate'].apply(lambda x: x.date())

    salesforce_data = salesforce_data[salesforce_data['lead_createdate'] <= salesforce_data['lead_close_date_impute']]

    date_suffix = dt.datetime.combine(dt.datetime.today(), dt.time.min).date().isoformat()

    salesforce_data.to_csv(output_path + "salesforce_" + str(date_suffix) + ".csv")
    return salesforce_data


def clean_ga_data(client, sql, output_path):
    """Transforms data for ga session data

    :return: dataframe
    """
    def landing_page_classification(a_path):
        if a_path.find('/blog') >= 0:
            return 'blog'
        elif a_path.find('/compare') >= 0:
            return "competitor"
        elif a_path.find("/sem") >= 0:
            return "sem"
        elif a_path.find("/brand") >= 0:
            return "brand"
        elif a_path.find("demo") >= 0:
            return "admin"
        elif a_path.find("admin") >= 0:
            return "admin"
        elif a_path.find("/aff") >= 0:
            return "affiliate"
        elif a_path.find("/lp") >= 0:
            return "display"
        elif a_path.find("/guides") >= 0:
            return "guides"
        else:
            return "site"

    ga_paths = utils.load_bigquery_data(client, sql)
    ga_paths = utils.standardize_columns(ga_paths)
    ga_paths = utils.convert_cols_to_datetime(ga_paths)
    ga_paths['session_date'] = ga_paths['date']

    ga_paths['landingpagepath'] = ga_paths['landingpagepath'].fillna('')
    ga_paths = ga_paths[~ga_paths['landingpagepath'].str.contains('store.volusion.com')]
    ga_paths['landingpagepath'] = ga_paths['landingpagepath'].apply(lambda x: url.urlparse(x)[2])

    ga_paths['campaign'] = ga_paths['campaign'].fillna('')
    ga_paths['non_brand'] = ga_paths['campaign'].apply(lambda x: 1 if x.find('-nbr-') else 0)
    ga_paths['landingpage_class'] = ga_paths['landingpagepath'].apply(landing_page_classification)
    ga_paths["landingpage_class"] = ga_paths["landingpage_class"].astype("category")

    ga_paths = ga_paths.join(pd.get_dummies(ga_paths["landingpage_class"]), how='left', rsuffix='_lp_ct')

    date_suffix = dt.datetime.combine(dt.datetime.today(), dt.time.min).date().isoformat()

    ga_paths.to_csv(output_path + "ga_sessions_" + str(date_suffix) + ".csv")
    return ga_paths


def merge_datasets(dataset, startdate, enddate, output_path):
    """merge intermediate datasets into one for feature extraction

    :param dataset: list of dataframes including salesforce_data, ga_data
    :param startdate: dataset start date
    :param enddate: end date of observations to include
    :param output_path: path to save file
    :return: merged dataframe for feature extraction
    """
    salesforce_df, ga_df = dataset[0], dataset[1]

    #merge sf and ga data
    paths_sf = salesforce_df.set_index("trial_order_detail_id").join(ga_df.set_index('demo_lookup'), rsuffix='_ga', how='left')
    paths_sf['date'] = paths_sf['date'].apply(lambda x: x.date())
    paths_sf = paths_sf[(paths_sf['date'] <= paths_sf['lead_close_date_impute']) | (paths_sf['date'].isnull())]

    session_counts = paths_sf.reset_index().groupby('index').count()['session_id']
    landingpage_counts = paths_sf.reset_index().groupby('index').sum()[['non_brand', u'admin',
                                                                        u'affiliate', 'blog',
                                                                        u'brand', u'competitor',
                                                                        'display', 'guides',
                                                                        'sem', 'site']]

    marketing_sources = ga_df[ga_df['first_demo_session'].notnull()]

    #master dataframe
    data_merged = salesforce_df.set_index("trial_order_detail_id").join(marketing_sources.set_index('demo_lookup'), how='left', rsuffix='_ms')

    data_merged = data_merged.join(session_counts, how='left', rsuffix='_count')
    data_merged = data_merged.join(landingpage_counts, how='left', rsuffix='_page_counts')
    data_merged['session_id_count'] = data_merged['session_id_count'].fillna(0)
    data_merged['session_count'] = data_merged['session_id_count']

    blog_sessions = ga_df[ga_df['landingpagepath'].str.contains("blog")].groupby('demo_lookup').count()['session_id']
    data_merged = data_merged.join(blog_sessions, how='left', rsuffix='_blog')
    data_merged['blog_sessions'] = data_merged['session_id_blog']
    return data_merged



def pca_transform(features, feature_names, ncomponents=50, output_path=''):
    """PCA transform dataset

    :param features: featureset to transform
    :param feature_names: feature names to transform
    :param ncomponents: number of principal components
    """

    #print(feature_names.shape)
    #print(features.shape)

    pca = IncrementalPCA(n_components=ncomponents, batch_size=50)
    pca.fit(features)
    log.info('{} variance explained by {} components.'.format(pca.explained_variance_ratio_.sum(), ncomponents))

    i = np.identity(features.shape[1])
    coef = pca.transform(i)

    date_suffix = dt.datetime.combine(dt.datetime.today(), dt.time.min).date().isoformat()

    feat_weights = pd.DataFrame(coef, columns=range(1, coef.shape[1]+1), index=feature_names[3:]).sort_values(1, ascending=False)
    feat_weights.to_csv(output_path+"pca_feat_weights_" + str(date_suffix) +".csv", sep=",")

    pca_X = pca.transform(features)
    np.save(output_path+'opp_PCA_X'+str(date_suffix), pca_X)
    return pca_X


def create_features(data, output_path, feature_names=None):
    """Create features from data"""

    # transformed_vars
    data['affiliate_touch'] = data['click_id'].notnull()
    data['affiliate_touch'] = data['affiliate_touch'].apply(lambda x: 1 if x else 0)
    t = ['True', 'Yes']
    data['have_products'] = data['have_products__c'].isin(t).apply(lambda x: 1 if x else 0)

    data['lead_day_of_week'] = data['lead_createdate'].apply(lambda x: x.weekday())

    def parse_industries(dataframe):
        i_list = []
        for x in dataframe['industry__c'].unique():
            if x:
                for ind in x.split(" / "):
                    i_list.append(ind)
                i_list = list(set(i_list))
        return i_list

    industry_list = parse_industries(data)
    for ind in industry_list:
        data[ind] = 0
        data.loc[(data["industry__c"].notnull()) & (data["industry__c"].str.contains(ind)), ind] = 1

    transformed_vars = ['converted_to_opp', 'blog_sessions', 'affiliate_touch', 'have_products', 'session_count',
                        'non_brand_page_counts', u'admin_page_counts', 'affiliate_page_counts', u'blog_page_counts',
                        u'brand_page_counts', u'competitor_page_counts', u'display_page_counts', u'guides_page_counts',
                        u'sem_page_counts', u'site_page_counts']

    transformed_vars.extend(industry_list)

    id_vars = ['trial_order_id', 'salesforce_id']

    # get dummy variables
    cat_variables = ['u_version', 'landingpage_class', 'devicecategory', 'medium',
                     u'socialnetwork', u'country_ms', u'lead_type__c', u'leadsource',
                     'device_type__c', u'number_of_products_selling__c', u'sms_opt_in__c',
                     'socialsignup__c', 'gender__c', 'have_products__c', 'position__c', 'lead_day_of_week',
                     u'hour_of_day']

    for x in cat_variables:
        data[x] = data[x].astype('category')

    fill = ['session_id_count', 'non_brand_page_counts', u'admin_page_counts',
            u'affiliate_page_counts', u'blog_page_counts', u'brand_page_counts',
            u'competitor_page_counts', u'display_page_counts',
            u'guides_page_counts', u'sem_page_counts', u'site_page_counts',
            u'session_count', u'session_id_blog', u'blog_sessions',
            u'affiliate_touch']

    for x in fill:
        data[x] = data[x].fillna(0)

    for t_var in transformed_vars:
        if t_var not in data.columns:
            data[t_var] = 0

    dummies = pd.get_dummies(data[cat_variables])
    joined_dummies = data[id_vars + transformed_vars].join(dummies)

    try:
        for feat in feature_names:
            if feat not in joined_dummies.columns:
                joined_dummies[feat] = 0
        joined_dummies = joined_dummies[feature_names]
    except TypeError:
        pass

    features_names = joined_dummies.columns
    features_set = joined_dummies[joined_dummies.columns[3:]].values
    target_variable = joined_dummies['converted_to_opp'].values
    id_list = joined_dummies[id_vars].values

    date_suffix = dt.datetime.combine(dt.datetime.today(), dt.time.min).date().isoformat()

    np.save(output_path+'id_list_'+ str(date_suffix), id_list)
    np.save(output_path+'opp_features_names_'+str(date_suffix), features_names)
    np.save(output_path+'opp_raw_features_X_'+str(date_suffix), features_set)
    np.save(output_path+'opp_target_Y_'+str(date_suffix), target_variable)

    return features_names, target_variable, features_set, id_list
