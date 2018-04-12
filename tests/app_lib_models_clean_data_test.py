"""test suite for data cleaning"""

from datetime import datetime as dt
import unittest

from src.lib.bigquery import bigquery as bq
from src.lib.models import clean_data

import src.config as conf
import src.data.queries as query


class CleanDataTests(unittest.TestCase):
    """UNIT TESTS MODELS PACKAGE"""
    config = conf.BaseConfig
    start_date = dt(2018, 2, 1).date()
    end_date = dt(2018, 3, 1).date()
    query_logic = query.QueryLogic

    def setUp(self):
        """Set up config variables"""
        salesforce_query = self.query_logic.SALEFORCE_QUERY.format(self.start_date, self.end_date)
        ga_query = self.query_logic.GA_QUERY.format(self.start_date, self.end_date)
        trial_conversions = self.query_logic.TRIAL_CONV_QUERY.format(self.start_date, self.end_date)
        gcp_project_name = self.config.BQ_PROJECT_ID
        dataset_name = self.config.LEADSCORING_DATASET
        salesforce_table = self.config.SALESFORCE_TABLE

        self.salesforce_query = salesforce_query
        self.ga_query = ga_query
        self.trial_conv = trial_conversions
        self.gcp_project = gcp_project_name
        self.dataset = dataset_name
        self.table_name = salesforce_table

    def test_clean_salesforce(self):
        """tests clean_salesforce method returns expected dataframe"""

        bq_client = bq.BigQueryClient(self.gcp_project, self.dataset, self.table_name)
        salesforce_data = clean_data.clean_salesforce_data(bq_client, self.salesforce_query)

        len_of_data = len(salesforce_data)
        expected_val = 13348

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

    def test_clean_ga(self):
        """tests clean_ga_data method returns expected dataframe"""

        bq_client = bq.BigQueryClient(self.gcp_project, self.dataset, self.table_name)
        ga_data = clean_data.clean_ga_data(bq_client, self.ga_query)

        len_of_data = len(ga_data)
        expected_val = 24996

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

    def test_clean_trial_conversions(self):
        """tests clean_trial_conversions returns expected dataframe"""

        bq_client = bq.BigQueryClient(self.gcp_project, self.dataset, self.table_name)
        trial_conversions = clean_data.clean_conversions_data(bq_client, self.trial_conv)

        len_of_data = len(trial_conversions)
        expected_val = 6246

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

    def test_merge_datasets(self):
        """tests merge data returns expected dataframe"""

        bq_client = bq.BigQueryClient(self.gcp_project, self.dataset, self.table_name)
        trial_conversions = clean_data.clean_conversions_data(bq_client, self.trial_conv)
        salesforce_data = clean_data.clean_salesforce_data(bq_client, self.salesforce_query)
        ga_data = clean_data.clean_ga_data(bq_client, self.ga_query)

        dataset = [salesforce_data, ga_data, trial_conversions]
        final_data = clean_data.merge_datasets(dataset, self.start_date, self.end_date, True)
        final_data_open = clean_data.merge_datasets(dataset, self.start_date, self.end_date, False)

        len_final_data = len(final_data)
        len_final_data_open = len(final_data_open)

        exp_val_data = 4613
        exp_val_data_open = 4424

        self.assertEqual(exp_val_data, len_final_data,
                         msg="After merging data,\
                          expected {}. Got {}".format(exp_val_data, len_final_data))

        self.assertEqual(exp_val_data_open, len_final_data_open,
                         msg="After merging data with open status,\
                          expected {}. Got {}".format(exp_val_data_open, len_final_data_open))

    def test_create_features(self):
        """tests create_feature function returns expected number of features and observations"""
        bq_client = bq.BigQueryClient(self.gcp_project, self.dataset, self.table_name)
        trial_conversions = clean_data.clean_conversions_data(bq_client, self.trial_conv)
        salesforce_data = clean_data.clean_salesforce_data(bq_client, self.salesforce_query)
        ga_data = clean_data.clean_ga_data(bq_client, self.ga_query)

        dataset = [salesforce_data, ga_data, trial_conversions]
        final_data = clean_data.merge_datasets(dataset, self.start_date, self.end_date, True)
        features_names, target_variable, _, _ = clean_data.create_features(final_data)

        feature_num = len(features_names)
        obs_count = len(target_variable)

        exp_feat_num = 1557
        exp_obs_count = 4614

        self.assertEqual(exp_feat_num, feature_num,
                         msg="After creating features,\
                          expected {} features. Got {}".format(exp_feat_num, feature_num))

        self.assertEqual(exp_obs_count, obs_count,
                         msg="After creating features,\
                          expected {} observations. Got {}".format(exp_obs_count, obs_count))


if __name__ == '__main__':
    unittest.main()
