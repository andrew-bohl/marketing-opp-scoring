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
        expected_val = 5345
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
        len_of_data = len(opps_data)
        expected_val = 1060
        print(len_of_data)

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

    def test_make_admin_data(self):
        """test make_admin_dataset method returns expected dataframe"""
        admin_data = clean_data.clean_admin_data(self.bq_client, self.admin_pv_query, "")
        opps_data = clean_data.clean_opps_data(self.bq_client, self.opps_query, "")
        X_opps_admin, Y_opps_admin = clean_data.make_admin_dataset(admin_data, opps_data)

        len_of_data = len(X_opps_admin)
        expected_val = 758

        num_converted = np.sum(Y_opps_admin[:, 0])
        num_converted_exp = 294
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
        opps_data = clean_data.clean_opps_data(self.bq_client, self.opps_query, "")

        X_opps_clicks, Y_opps_clicks= clean_data.make_v2click_dataset(v2clicks_data, opps_data)

        len_of_data = len(X_opps_clicks)
        expected_val = 392

        num_converted = np.sum(Y_opps_clicks[:, 0])
        num_converted_exp = 177

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

        self.assertEqual(num_converted_exp, num_converted,
                         msg="After cleaning data,\
                          expected {} got {}".format(num_converted_exp, num_converted))

    def test_make_ga_data(self):
        """test make_admin_dataset method returns expected dataframe"""
        ga_data = clean_data.clean_ga_data(self.bq_client, self.ga_query, "")
        opps_data = clean_data.clean_opps_data(self.bq_client, self.opps_query, "")

        X_opps_ga, Y_opps_ga= clean_data.make_ga_dataset(ga_data, opps_data)

        len_of_data = len(X_opps_ga)
        expected_val = 1436

        num_converted = np.sum(Y_opps_ga[:, 0])
        num_converted_exp = 633

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

        self.assertEqual(num_converted_exp, num_converted,
                         msg="After cleaning data,\
                          expected {} got {}".format(num_converted_exp, num_converted))

    def test_make_leads_data(self):
        """test make_leads_dataset method returns expected dataframe"""
        leads_data = clean_data.clean_leads_data(self.bq_client, self.salesforce_query, "")

        X_opps_leads, Y_opps_leads = clean_data.make_leads_dataset(leads_data)

        len_of_data = len(X_opps_leads)
        expected_val = 883

        num_converted = np.sum(Y_opps_leads[:, 0])
        num_converted_exp = 177

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

        self.assertEqual(num_converted_exp, num_converted,
                         msg="After cleaning data,\
                          expected {} got {}".format(num_converted_exp, num_converted))

    def test_make_tasks_data(self):
        """tests make_tasks_dataset method returns expected dataset"""
        tasks_data = clean_data.clean_tasks_data(self.bq_client, self.tasks_query, "")
        opps_data = clean_data.clean_opps_data(self.bq_client, self.opps_query, "")

        X_opps_task, Y_opps_task= clean_data.make_tasks_dataset(tasks_data, opps_data)

        len_of_data = len(X_opps_task)
        expected_val = 82

        num_converted = np.sum(Y_opps_task[:, 0])
        num_converted_exp = 24

        self.assertEqual(expected_val, len_of_data,
                         msg="After cleaning data,\
                          expected {} got {}".format(expected_val, len_of_data))

        self.assertEqual(num_converted_exp, num_converted,
                         msg="After cleaning data,\
                          expected {} got {}".format(num_converted_exp, num_converted))


if __name__ == '__main__':
    unittest.main()
