"""  DATA CLEANING MODULE
    Loads and preps data for modeling
"""
import datetime as dt

import pandas as pd
import urllib.parse as url

from src.data import filters
import src.lib.models.utilities as utils


date_suffix = dt.datetime.combine(dt.datetime.today(), dt.time.min).date().isoformat()


def clean_admin_data(client, sql):
    """Cleans the opportunity conversion data

    :param client: BQ client
    :param sql: sql query
    :param output_path: where to save file
    :return: cleaned opp data
    """
    admin_data = utils.load_bigquery_data(client, sql)
    admin_data['inserted_at'] = pd.to_datetime(admin_data['inserted_at'])
    admin_data = admin_data[admin_data['url'].notnull()]
    admin_data['url_path'] = admin_data['url'].apply(lambda x: x[x.find('.com')+5:])
    admin_data['url_path'] = admin_data['url_path'].apply(lambda x: x[:x.find('/')+1])
    admin_data.loc[admin_data['url_path'] == '', 'url_path'] = admin_data[admin_data['url_path'] == '']['url'].apply(lambda x: x[x.find('/')+1:])
    admin_data['url_path'] = admin_data['url_path'].apply(lambda x: x.replace('/', ''))
    admin_data = admin_data[~admin_data['url_path'].str.contains('localhost')]
    admin_data = admin_data[~admin_data['url_path'].str.contains('admin-dev')]

    unique_urlpaths = admin_data.groupby('url_path').count()['pid'].reset_index().sort_values('pid', ascending=False)
    top_urls = unique_urlpaths['url_path'].head(8).tolist()

    admin_data['url_path_2'] = admin_data['url_path'].apply(lambda x: x if x in top_urls else 'Other')
    admin_data = admin_data.join(pd.get_dummies(admin_data['url_path_2']))
    return admin_data


def clean_v2clicks_data(client, sql):
    """Cleans v2clicks data for model

    :param client: BQ client
    :param sql: sql query
    :param output_path: where to save file
    :return: cleaned opp data
    """
    v2_clicks_data = utils.load_bigquery_data(client, sql)
    v2_clicks_data['inserted_at'] = pd.to_datetime(v2_clicks_data['inserted_at'])
    v2_clicks_data = v2_clicks_data.join(pd.get_dummies(v2_clicks_data['action']))
    # v2_clicks_data.to_csv(output_path + "v2_clicks_data_" + str(date_suffix) + ".csv")
    return v2_clicks_data


def clean_tasks_data(client, sql):
    """Cleans salesforce tasks data for merging
    :param client: BQ client
    :param sql: sql query
    :param output_path: where to save file
    :return: cleaned tasks data
    """

    tasks_data = utils.load_bigquery_data(client, sql)
    task_values = tasks_data.groupby('Activity_Type__c').count()['task_id'].reset_index().sort_values('task_id', ascending=False)
    top_tasks = task_values['Activity_Type__c'].head(6).tolist()
    tasks_data['activity_type_2'] = tasks_data['Activity_Type__c'].apply(lambda x: x if x in top_tasks else 'Other')
    tasks_data['TaskSubtype'] = tasks_data['TaskSubtype'].apply(lambda x: x+ "_sub")

    tasks_data = tasks_data.join(pd.get_dummies(tasks_data[['activity_type_2', 'TaskSubtype']]))
    tasks_data['trial_id'] = tasks_data['order_detail_id'].combine_first(tasks_data['tenant_id__c'])
    return tasks_data


def clean_ga_data(client, sql):
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
    ga_paths = ga_paths[ga_paths['landingpagepath'].notnull()]
    ga_paths = ga_paths[~ga_paths['landingpagepath'].str.contains('store.volusion.com')]
    ga_paths['landingpagepath'] = ga_paths['landingpagepath'].apply(lambda x: url.urlparse(x)[2])

    ga_paths['landingpagepath'] = ga_paths['landingpagepath']\
        .apply(lambda x: 'www.volusion.com/login' if x.find('login') >= 0 else x)
    ga_paths['landingpagepath'] = ga_paths['landingpagepath'].apply(lambda x: x.replace('/v2', ''))
    ga_paths['landingpagepath'] = ga_paths['landingpagepath'].apply(lambda x: x.replace('/v1', ''))

    top_landingpages = ga_paths[['landingpagepath']].apply(pd.value_counts)\
        .sort_values('landingpagepath', ascending=False).head(20).reset_index()['index'].tolist()
    ga_paths['landingpagepath_2'] = ga_paths['landingpagepath'].apply(lambda x: x if x in top_landingpages else 'Other')

    ga_paths['campaign'] = ga_paths['campaign'].fillna('')
    ga_paths['non_brand'] = ga_paths['campaign'].apply(lambda x: 1 if x.find('-nbr-') else 0)
    ga_paths['landingpage_class'] = ga_paths['landingpagepath'].apply(landing_page_classification)
    ga_paths['date'] = pd.to_datetime(ga_paths['date'])
    return ga_paths


def clean_opps_data(client, sql):
    """Transforms data for salesforce

    :return: SalesForce dataframe
    """
    opps_data = utils.load_bigquery_data(client, sql)
    opps_data['lead_createdate'] = pd.to_datetime(opps_data['lead_createdate'])
    return opps_data


def clean_leads_data(client, sql):
    """Clean salesforce data for model

    :return: SalesForce dataframe
    """
    def parse_industries(dataframe):
        i_list = []
        for x in dataframe['industry__c'].unique():
            if x:
                for industry in x.split(" / "):
                    i_list.append(industry)
                i_list = list(set(i_list))
        return i_list

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

    # filters out ids not of len 7

    salesforce_data = filter_bad_orderids(salesforce_data, 'trial_order_id', 7)
    salesforce_data = salesforce_data.sort_values(['trial_order_id', 'lead_createdate'], ascending=False) \
        .groupby('trial_order_id').first().reset_index()
    salesforce_data = utils.convert_cols_to_datetime(salesforce_data)

    top_country = salesforce_data.groupby('country').count()['trial_order_id'].reset_index() \
        .sort_values('trial_order_id', ascending=False)['country'].head(5).tolist()
    salesforce_data['country_2'] = salesforce_data['country'].apply(lambda x: x if x in top_country else 'Other')

    # create hour of day feature
    salesforce_data["hour_of_day"] = salesforce_data["lead_createdate"].apply(lambda x: x.hour)

    leadowners = salesforce_data[['lead_owner_full_name__c']].apply(pd.value_counts) \
        .sort_values('lead_owner_full_name__c', ascending=False).reset_index().head(30)['index'].tolist()
    salesforce_data['leadowners_2'] = salesforce_data['lead_owner_full_name__c'] \
        .apply(lambda x: x if x in leadowners else 'Other')

    salesforce_data['created_date_dt'] = salesforce_data['lead_createdate'].apply(lambda x: x.date())

    salesforce_data['industry__c'] = salesforce_data['industry__c'].fillna('NA')
    industry_list = parse_industries(salesforce_data)

    for ind in industry_list:
        salesforce_data[ind] = 0
        salesforce_data.loc[(salesforce_data["industry__c"].notnull())
                            & (salesforce_data["industry__c"].str.contains(ind)), ind] = 1

    lead_types_list = salesforce_data.groupby('lead_type__c').count()['trial_order_id'] \
        .sort_values(ascending=False).head(10).reset_index()['lead_type__c'].tolist()
    salesforce_data['lead_type_2'] = salesforce_data['lead_type__c'] \
        .apply(lambda x: x if x in lead_types_list else 'Other')
    return salesforce_data


def make_admin_dataset(dataset, feature_names=None, training=True):
    """Creates the feature set for the first model"""
    admin_data, opps_data = dataset[0], dataset[1]
    opps_admin = pd.merge(opps_data, admin_data, left_on='trial_order_detail_id', right_on='u_id', how='inner')
    opps_admin = opps_admin.join(pd.get_dummies(opps_admin['version']))
    opps_admin['cut_off'] = opps_admin['lead_createdate'].apply(lambda x: x.date() + dt.timedelta(days=21))
    opps_admin['opp_createdate'] = pd.to_datetime(opps_admin[opps_admin['opp_createdate'].notnull()]['opp_createdate'])
    opps_admin['opp_createdate'] = opps_admin['opp_createdate'].apply(lambda x: x.date())

    opps_admin['inserted_at_date'] = opps_admin['inserted_at'].apply(lambda x: x.date())

    if training:
        opps_admin['cut_off'] = opps_admin['opp_createdate'].combine_first(opps_admin['cut_off'])
        opps_admin = opps_admin[opps_admin['inserted_at_date'] <= opps_admin['cut_off']]

    opps_mdl_data = opps_admin.groupby('trial_order_detail_id').sum()

    opps_mdl_data['converted_to_opp'] = opps_mdl_data['converted_to_opp'].apply(lambda x: 1 if x > 0 else 0)

    opps_mdl_data['v1'] = opps_mdl_data['v1'].apply(lambda x: 1 if x > 0 else 0)
    opps_mdl_data['v2'] = opps_mdl_data['v2'].apply(lambda x: 1 if x > 0 else 0)

    top_urls = admin_data['url_path_2'].unique().tolist()
    top_urls.append('v1')
    top_urls.append('v2')
    top_urls.append('converted_to_opp')

    opps_mdl_data = opps_mdl_data[top_urls].reset_index()
    opps_mdl_data = opps_mdl_data.fillna(0)

    if feature_names:
        for feat in feature_names['admin']:
            try:
                opps_mdl_data[feat]
            except KeyError:
                opps_mdl_data[feat] = 0.5
        feat_cols = feature_names['admin']
        opps_mdl_data_norm = (opps_mdl_data[feat_cols] - opps_mdl_data[feat_cols].mean()) / (
                opps_mdl_data[feat_cols].max() - opps_mdl_data[feat_cols].min())
    else:
        opps_mdl_data_norm = (opps_mdl_data[opps_mdl_data.columns[1:-1]] - opps_mdl_data[
            opps_mdl_data.columns[1:-1]].mean()) / (opps_mdl_data[opps_mdl_data.columns[1:-1]].max()\
                                                    - opps_mdl_data[opps_mdl_data.columns[1:-1]].min())

    X_opps_admin = opps_mdl_data_norm.values
    Y_opps_admin = opps_mdl_data[['converted_to_opp', 'trial_order_detail_id']].values

    return X_opps_admin, Y_opps_admin, opps_mdl_data_norm.columns


def make_v2click_dataset(dataset, feature_names=None, training=True):
    """Make dataset for v2clicks model"""

    v2_clicks_data, opps_data = dataset[0], dataset[1]
    feats = v2_clicks_data['action'].unique().tolist()
    opps_v2clicks = pd.merge(opps_data, v2_clicks_data, left_on='trial_order_detail_id',
                             right_on='tenantId',
                             how='inner')

    opps_v2clicks['opp_createdate'] = pd.to_datetime(opps_v2clicks[opps_v2clicks['opp_createdate'].notnull()]['opp_createdate'])
    opps_v2clicks['opp_createdate'] = opps_v2clicks['opp_createdate'].apply(lambda x: x.date())

    opps_v2clicks['inserted_at_date'] = opps_v2clicks['inserted_at'].apply(lambda x: x.date())

    if training:
        opps_v2clicks['cut_off'] = opps_v2clicks['lead_createdate'].apply(lambda x: x.date() + dt.timedelta(days=21))
        opps_v2clicks['cut_off'] = opps_v2clicks['opp_createdate'].combine_first(opps_v2clicks['cut_off'])
        opps_v2clicks = opps_v2clicks[opps_v2clicks['inserted_at_date'] <= opps_v2clicks['cut_off']]

    opps_v2clicks = opps_v2clicks.groupby('trial_order_detail_id').sum().reset_index()

    opps_v2clicks['converted_to_opp'] = opps_v2clicks['converted_to_opp'].apply(lambda x: 1 if x > 0 else 0)

    if feature_names:
        for featx in feature_names['v2clicks']:
            try:
                opps_v2clicks[featx]
            except KeyError:
                opps_v2clicks[featx] = 0.5
        feat_cols = feature_names['v2clicks']
        opps_v2clicks_norm = (opps_v2clicks[feat_cols] - opps_v2clicks[feat_cols].mean()) / (
                    opps_v2clicks[feat_cols].max() - opps_v2clicks[feat_cols].min())
    else:
        opps_v2clicks_norm = (opps_v2clicks[feats] - opps_v2clicks[feats].mean()) / (
                opps_v2clicks[feats].max() - opps_v2clicks[feats].min())

    X_opps_v2clicks = opps_v2clicks_norm.values
    Y_opps_v2clicks = opps_v2clicks[['converted_to_opp', 'trial_order_detail_id']].values

    return X_opps_v2clicks, Y_opps_v2clicks, opps_v2clicks_norm.columns


def make_tasks_dataset(dataset, feature_names=None, training=True):
    """make dataset for tasks model"""

    tasks_data, opps_data = dataset[0], dataset[1]
    tasks_data_opps = pd.merge(opps_data, tasks_data, left_on='trial_order_detail_id', right_on='trial_id', how='inner')

    tasks_data_opps['lead_createdate'] = pd.to_datetime(tasks_data_opps['lead_createdate'])
    tasks_data_opps['cut_off'] = tasks_data_opps['lead_createdate'].apply(lambda x: x.date() + dt.timedelta(days=21))
    tasks_data_opps['opp_createdate'] = pd.to_datetime(tasks_data_opps['opp_createdate']).apply(lambda x: x.date())
    tasks_data_opps['ActivityDate'] = pd.to_datetime(tasks_data_opps['ActivityDate']).apply(lambda x: x.date())

    if training:
        tasks_data_opps['cut_off'] = tasks_data_opps['opp_createdate'].combine_first(tasks_data_opps['cut_off'])
        tasks_data_opps = tasks_data_opps[tasks_data_opps['ActivityDate'] <= tasks_data_opps['cut_off']]

    tasks_data_opps = tasks_data_opps.groupby('trial_id').sum().reset_index()

    tasks_data_opps['converted_to_opp'] = tasks_data_opps['converted_to_opp'].apply(lambda x: 1 if x > 0 else 0)
    feats = tasks_data_opps.columns

    for col in feats[2:]:
        tasks_data_opps[col] = tasks_data_opps[col].fillna(tasks_data_opps[col].mean())

    if feature_names:
        for feat in feature_names['tasks']:
            try:
                tasks_data_opps[feat]
            except KeyError:
                tasks_data_opps[feat] = 0.5
        feat_cols = feature_names['tasks']
        tasks_data_opps_norm = (tasks_data_opps[feat_cols] - tasks_data_opps[feat_cols].mean()) / \
                               (tasks_data_opps[feat_cols].max() - tasks_data_opps[feat_cols].min())

    else:
        tasks_data_opps_norm = (tasks_data_opps[feats[2:]] - tasks_data_opps[feats[2:]].mean()) / \
                               (tasks_data_opps[feats[2:]].max() - tasks_data_opps[feats[2:]].min())

    tasks_data_opps_norm = tasks_data_opps_norm.fillna(0)

    X_opps_tasks = tasks_data_opps_norm.values
    Y_opps_tasks = tasks_data_opps[['converted_to_opp', 'trial_id']].values

    return X_opps_tasks, Y_opps_tasks, tasks_data_opps_norm.columns


def make_leads_dataset(salesforce_data, feature_names=None):
    """Transforms data for salesforce for model

    :return: SalesForce dataframe
    """

    def parse_industries(dataframe):
        i_list = []
        for x in dataframe['industry__c'].unique():
            if x:
                for industry in x.split(" / "):
                    i_list.append(industry)
                i_list = list(set(i_list))
        return i_list

    industry_list = parse_industries(salesforce_data)
    dummies_col_lts = ['lead_type_2', 'leadsource', 'leadowners_2', 'device_type__c', 'gender__c', 'position__c',
                       'country_2']

    cols = []
    cols.extend(industry_list)
    cols.append('trial_order_detail_id')
    cols.append('converted_to_opp')

    cols.reverse()

    salesforce_data = salesforce_data[cols].join(pd.get_dummies(salesforce_data[dummies_col_lts]))
    salesforce_data['converted_to_opp'] = salesforce_data['converted_to_opp'].apply(lambda x: int(x))

    feat_set = salesforce_data.columns

    if feature_names:
        for feat in feature_names['sf_leads']:
            try:
                salesforce_data[feat]
            except KeyError:
                salesforce_data[feat] = 0.5

        feat_cols = feature_names['sf_leads']
        opps_norm = (salesforce_data[feat_cols] - salesforce_data[feat_cols].mean()) / (
            salesforce_data[feat_cols].max() - salesforce_data[feat_cols].min())

    else:
        opps_norm = (salesforce_data[feat_set[2:]] - salesforce_data[feat_set[2:]].mean()) / (
            salesforce_data[feat_set[2:]].max() - salesforce_data[feat_set[2:]].min())

    salesforce_data['converted_to_opp'] = salesforce_data['converted_to_opp'].fillna(0)

    X_opps_sf = opps_norm.values
    Y_opps_sf = salesforce_data[['converted_to_opp', 'trial_order_detail_id']].values

    return X_opps_sf, Y_opps_sf, opps_norm.columns


def make_ga_dataset(dataset, feature_names=None, training=True):
    """Make dataset for GA data model
    :param dataset: is a tuple of ga_paths, opps_data
    :param feature_names: is a list of featurenames
    :param training: Boolean to filter on a cutoff date"""

    ga_paths, opps_data = dataset[0], dataset[1]

    opps_paths = pd.merge(ga_paths, opps_data, left_on='demo_lookup', right_on='trial_order_detail_id')
    opps_paths['opp_createdate'] = pd.to_datetime(opps_paths['opp_createdate'])
    opps_paths['opp_createdate'] = opps_paths['opp_createdate'].apply(lambda x: x.date())
    opps_paths['date'] = pd.to_datetime(opps_paths['date']).apply(lambda x: x.date())

    if training:
        opps_paths['cut_off'] = opps_paths['lead_createdate'].apply(lambda x: x.date() + dt.timedelta(days=21))
        opps_paths['cut_off'] = opps_paths['opp_createdate'].combine_first(opps_paths['cut_off'])
        opps_paths = opps_paths[opps_paths['date'] <= opps_paths['cut_off']]

    cols = ['trial_order_detail_id', 'converted_to_opp']
    feats = ['landingpage_class', 'devicecategory', 'brand_nonbrand', 'landingpagepath_2', 's_version']
    opps_path_feats = opps_paths[cols].join(pd.get_dummies(opps_paths[feats]))
    feat_set = opps_path_feats.columns

    for col in feat_set[2:]:
        opps_path_feats[col] = opps_path_feats[col].fillna(opps_path_feats[col].mean())

    if feature_names:
        for feat in feature_names['ga']:
            try:
                opps_path_feats[feat]
            except KeyError:
                opps_path_feats[feat] = 0.5
        feat_cols = feature_names['ga']

        opps_path_feats_norm = (opps_path_feats[feat_cols] - opps_path_feats[feat_cols].mean()) / (
                    opps_path_feats[feat_cols].max() - opps_path_feats[feat_cols].min())

    else:
        opps_path_feats_norm = (opps_path_feats[feat_set[2:]] - opps_path_feats[feat_set[2:]].mean()) / (
                    opps_path_feats[feat_set[2:]].max() - opps_path_feats[feat_set[2:]].min())

    opps_path_feats['converted_to_opp'] = opps_path_feats['converted_to_opp'].fillna(0)

    X_opps_ga = opps_path_feats_norm.values
    Y_opps_ga = opps_path_feats[['converted_to_opp', 'trial_order_detail_id']].values

    return X_opps_ga, Y_opps_ga, opps_path_feats_norm.columns


