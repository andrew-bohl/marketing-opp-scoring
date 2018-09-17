"""test suite for data cleaning"""

from datetime import datetime as dt
import os
import unittest

import numpy as np

from src.lib import bigquery as bq
from src.lib.models import clean_data
from src.main import app
import src.data.queries as query


class CleanDataTests(unittest.TestCase):
    """UNIT TESTS MODELS PACKAGE"""
    def setUp(self):
        """Set up config variables"""
        self.config = app.config
        start_date = dt(2018, 8, 1).date()
        end_date = dt(2018, 8, 4).date()
        query_logic = query.QueryLogic

        salesforce_query = query_logic.SALEFORCE_QUERY.format(start_date, end_date)
        ga_query = query_logic.GA_QUERY.format(start_date, end_date)
        opps_query = query_logic.OPPS_QUERY.format(start_date, end_date)
        admin_pv_query = query_logic.ADMINPV_QUERY.format(start_date, end_date)
        v2clicks_query = query_logic.V2Clicks_QUERY.format(start_date, end_date)
        tasks_query = query_logic.SFTASKS_QUERY.format(start_date, end_date)

        gcp_project_name = self.config["BQ_PROJECT_ID"]
        dataset_name = self.config["LEADSCORING_DATASET"]

        self.opps_query = opps_query
        self.admin_pv_query = admin_pv_query
        self.v2clicks_query = v2clicks_query
        self.tasks_query = tasks_query
        self.salesforce_query = salesforce_query
        self.ga_query = ga_query

        self.gcp_project = gcp_project_name
        self.dataset = dataset_name
        self.table_name = self.config["SALESFORCE_TABLE"]
        self.relative_path = os.path.dirname(__file__)

        self.opps_data = None

        bq_client = bq.BigQueryClient(self.gcp_project,
                                      self.dataset,
                                      self.table_name,
                                      f'{self.relative_path}/credentials/leadscoring.json')

        self.bq_client = bq_client

    def test_clean_leads(self):
        """tests clean_salesforce method returns expected dataframe"""

        salesforce_data = clean_data.clean_leads_data(self.bq_client, self.salesforce_query, "")

        len_of_data = len(salesforce_data)
        expected_val = 883
        print(len_of_data)

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

    def test_clean_ga(self):
        """tests clean_ga_data method returns expected dataframe"""

        ga_data = clean_data.clean_ga_data(self.bq_client, self.ga_query, "")
        len_of_data = len(ga_data)
        expected_val = 5333
        print(len_of_data)

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

    def test_clean_admin(self):
        """tests clean_ga_data method returns expected dataframe"""
        admin_data = clean_data.clean_admin_data(self.bq_client, self.admin_pv_query, "")
        len_of_data = len(admin_data)
        expected_val = 66225
        print(len_of_data)

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

    def test_clean_v2clicks(self):
        """tests clean_ga_data method returns expected dataframe"""
        click_data = clean_data.clean_v2clicks_data(self.bq_client, self.v2clicks_query, "")
        len_of_data = len(click_data)
        expected_val = 2874
        print(len_of_data)

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

    def test_clean_tasks(self):
        """tests clean_ga_data method returns expected dataframe"""
        tasks_data = clean_data.clean_tasks_data(self.bq_client, self.tasks_query, "")
        len_of_data = len(tasks_data)
        expected_val = 3695
        print(len_of_data)

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

    def test_clean_opps(self):
        opps_data = clean_data.clean_opps_data(self.bq_client, self.opps_query, "")
        self.opps_data = opps_data
        len_of_data = len(opps_data)
        expected_val = 1060
        print(len_of_data)

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

    def test_make_admin_data(self):
        """test make_admin_dataset method returns expected dataframe"""
        admin_data = clean_data.clean_admin_data(self.bq_client, self.admin_pv_query, "")
        X_opps_admin, Y_opps_admin = clean_data.make_admin_dataset(admin_data, self.opps_data)

        len_of_data = len(X_opps_admin)
        expected_val = 3867

        num_converted = np.sum(Y_opps_admin)
        num_converted_exp = 231
        print(len_of_data)

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

        self.assertEqual(num_converted_exp, num_converted,
                         msg="After cleaning data,\
                          expected {} got {}".format(num_converted_exp, num_converted))

    def test_make_v2clicks_data(self):
        """test make_admin_dataset method returns expected dataframe"""
        v2clicks_data = clean_data.clean_v2clicks_data(self.bq_client, self.v2clicks_query, "")
        X_opps_clicks, Y_opps_clicks= clean_data.make_v2click_dataset(v2clicks_data, self.opps_data)

        len_of_data = len(X_opps_clicks)
        expected_val = 3867

        num_converted = np.sum(Y_opps_clicks)
        num_converted_exp = 231
        print(len_of_data)

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

        self.assertEqual(num_converted_exp, num_converted,
                         msg="After cleaning data,\
                          expected {} got {}".format(num_converted_exp, num_converted))

    def test_make_ga_data(self):
        """test make_admin_dataset method returns expected dataframe"""
        ga_data = clean_data.clean_ga_data(self.bq_client, self.ga_query, "")
        X_opps_ga, Y_opps_clicks= clean_data.make_ga_dataset(ga_data, self.opps_data)

        len_of_data = len(X_opps_ga)
        expected_val = 3867

        num_converted = np.sum(Y_opps_clicks)
        num_converted_exp = 231
        np.sum()
        print(len_of_data)

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

        self.assertEqual(num_converted_exp, num_converted,
                         msg="After cleaning data,\
                          expected {} got {}".format(num_converted_exp, num_converted))



    # def test_merge_datasets(self):
    #     """tests merge data returns expected dataframe"""
    #
    #     salesforce_data = clean_data.clean_salesforce_data(self.bq_client, self.salesforce_query)
    #     ga_data = clean_data.clean_ga_data(self.bq_client, self.ga_query)
    #
    #     dataset = [salesforce_data, ga_data]
    #     final_data = clean_data.merge_datasets(dataset, self.start_date, self.end_date, True)
    #     final_data_open = clean_data.merge_datasets(dataset, self.start_date, self.end_date, False)
    #
    #     len_final_data = len(final_data)
    #     len_final_data_open = len(final_data_open)
    #
    #     exp_val_data = 4613
    #     exp_val_data_open = 4424
    #
    #     self.assertEqual(exp_val_data, len_final_data,
    #                      msg="After merging data,\
    #                       expected {}. Got {}".format(exp_val_data, len_final_data))
    #
    #     self.assertEqual(exp_val_data_open, len_final_data_open,
    #                      msg="After merging data with open status,\
    #                       expected {}. Got {}".format(exp_val_data_open, len_final_data_open))
    #
    # def test_create_features(self):
    #     """tests create_feature function returns expected number of features and observations"""
    #
    #     salesforce_data = clean_data.clean_salesforce_data(self.bq_client, self.salesforce_query)
    #     ga_data = clean_data.clean_ga_data(self.bq_client, self.ga_query)
    #
    #     dataset = [salesforce_data, ga_data]
    #     final_data = clean_data.merge_datasets(dataset, self.start_date, self.end_date, True)
    #     features_names, target_variable, _, _ = clean_data.create_features(final_data)
    #
    #     feature_num = len(features_names)
    #     obs_count = len(target_variable)
    #
    #     exp_feat_num = 1557
    #     exp_obs_count = 4614
    #
    #     self.assertEqual(exp_feat_num, feature_num,
    #                      msg="After creating features,\
    #                       expected {} features. Got {}".format(exp_feat_num, feature_num))
    #
    #     self.assertEqual(exp_obs_count, obs_count,
    #                      msg="After creating features,\
    #                       expected {} observations. Got {}".format(exp_obs_count, obs_count))


if __name__ == '__main__':
    unittest.main()
