"""test suite for data loading and model evaluation"""

from datetime import datetime as dt
import logging as log
import numpy as np
import os
import unittest

from src.main import app
from src.lib.models import models


class ModelTests(unittest.TestCase):
    """UNIT TESTS MODELS PACKAGE"""

    def setUp(self):
        """set up model for testing"""
        self.model = models.Model(app.config)

        output_path = self.model.config["OUTPUTS_PATH"]
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.start_date = dt(2017, 8, 1).date()
        self.end_date = dt(2018, 8, 4).date()

        opps_data, ga_paths, tasks_data, v2_clicks_data, admin_data, salesforce_data = self.model.load_data(self.start_date, self.end_date)
        self.salesforce = salesforce_data
        self.ga_data = ga_paths
        self.opps_data = opps_data
        self.tasks_data = tasks_data
        self.v2_clicks_data = v2_clicks_data
        self.admin_data = admin_data

    def test_load_data(self):
        """Tests load data function is generating approx size data"""
        salesforce_data = self.salesforce
        ga_paths = self.ga_data
        sf_data_sz = len(salesforce_data)
        ga_data_sz = len(ga_paths)

        self.assertEqual(42700, round(sf_data_sz, -2),
                         msg="Salesforce data size was not as expected. Got {}, expected around 42700".format(sf_data_sz))
        self.assertEqual(134300, round(ga_data_sz, -2),
                         msg="GA data size was not as expected. Got {}, expected around 134300".format(ga_data_sz))

    def test_dataset_size(self):
        """tests dataset is around 31000 observations

        Asserts: dataset length is around 32K
        """
        salesforce_data = self.salesforce
        ga_paths = self.ga_data
        datasets = [salesforce_data, ga_paths]

        self.model.create_model_data(datasets, self.start_date, self.end_date)

        size = len(self.model.target)
        #check that data set is accurate
        self.assertTrue((30000 <= size) & (size <= 32000),
                        msg="Data set size not between [30000, 32000], expected range, got {}".format(size))

    def test_score_set(self):
        """Tests create score set"""
        salesforce_data = self.salesforce
        ga_paths = self.ga_data
        datasets = [salesforce_data, ga_paths]

        self.model.create_score_set(datasets, self.start_date, self.end_date)

        size = len(self.model.to_score)

        self.assertTrue((26000 <= size) & (size <= 27000),
                        msg="Data set size not within expected range, got {}".format(size))

    def test_split_dataset(self):
        """"Test split size"""
        salesforce_data = self.salesforce
        ga_paths = self.ga_data
        datasets = [salesforce_data, ga_paths]

        self.model.create_model_data(datasets, self.start_date, self.end_date)

        features = self.model.features
        target = self.model.target

        self.model.split_dataset(features, target, test_size=0.4, rd_state=4)

        train_set = self.model.train_set
        test_set = self.model.test_set
        feature_len = len(self.model.feat_names)
        log.info("features used in model: %d" % feature_len)
        log.info("training size in model: %d" % len(train_set[0]))
        log.info("testing size in model: %d" % len(test_set[0]))

        self.assertEqual(2814, feature_len,
                         msg="Test set size was different than expected, got {}".format(feature_len))

        self.assertEqual(12000, round(len(test_set[0]), -3),
                         msg="Test set size was different than expected, got {}".format(round(len(test_set[0]), -3)))

        self.assertEqual(6400, round(len(train_set[0]), -2),
                         msg="Test set size was different than expected, got {}".format(len(train_set[0])))

    def test_model_AUC(self):
        """
        Tests that model trained on balanced dataset has test AUC of 0.77, train AUC of 0.83
        Asserts: test AUC = 0.77, train AUC = 0.83
        """

        x_data = np.load('test_data/train_X.npy')
        y_data = np.load('test_data/train_Y.npy')
        train_set = [y_data, x_data]

        test_model = models.Model()
        test_model.train_set = train_set
        test_model.train_model(train_set)

        x_test = np.load("test_data/test_X.npy")
        y_test = np.load("test_data/test_Y.npy")
        test_set = [y_test, x_test]
        test_model.evaluate_model(test_set, train_set)
        print(test_model.evaluation_metrics)

        train_auc = round(test_model.evaluation_metrics['train_AUC'], 2)
        test_auc = round(test_model.evaluation_metrics['test_AUC'], 2)

        self.assertEqual(0.85, train_auc,
                         msg="Train AUC not expected. Got {}, expected 0.85".format(train_auc))

        self.assertEqual(0.77, test_auc,
                         msg="Test AUC not expected. Got {}, expected 0.77".format(test_auc))


if __name__ == '__main__':
    unittest.main()
