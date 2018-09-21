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

        self.start_date = dt(2018, 8, 3).date()
        self.end_date = dt(2018, 8, 14).date()

        opps_data, ga_paths, tasks_data, v2_clicks_data, admin_data, salesforce_data = self.model.load_data(self.start_date, self.end_date)
        self.salesforce = salesforce_data
        self.ga_data = ga_paths
        self.opps_data = opps_data
        self.tasks_data = tasks_data
        self.v2_clicks_data = v2_clicks_data
        self.admin_data = admin_data

        ids, datasets = self.model.create_model_data(self.start_date, self.end_date, training=True)
        self.ids = ids
        self.datasets = datasets

        self.model.split_dataset(ids, datasets, test_size=0.3)
        self.model.create_logreg_models(self.model.train_set)

        train_features, train_y, _ = models.create_ensemble_features(self.model.models, self.model.train_set, self.ids)
        self.train_features = train_features
        self.train_y = train_y
        test_features, test_y, _ = models.create_ensemble_features(self.model.models, self.model.test_set, self.ids)
        self.test_features = test_features
        self.test_y = test_y

    def test_load_data(self):
        """Tests load data function is generating approx size data"""
        sf_data_sz = len(self.salesforce)
        ga_data_sz = len(self.ga_data)
        clicks_data_sz = len(self.v2_clicks_data)
        tasks_data_sz = len(self.tasks_data)
        admin_data_sz = len(self.admin_data)
        opps_data_sz = len(self.opps_data)

        self.assertEqual(42700, round(sf_data_sz, -2),
                         msg="Salesforce data size was not as expected. Got {}, expected around 42700".format(sf_data_sz))
        self.assertEqual(42700, round(ga_data_sz, -2),
                         msg="GA data size was not as expected. Got {}, expected around 42700".format(ga_data_sz))
        self.assertEqual(42700, round(clicks_data_sz, -2),
                         msg="v2 clicks data size was not as expected. Got {}, expected around 42700".format(clicks_data_sz))
        self.assertEqual(42700, round(tasks_data_sz, -2),
                         msg="tasks data size was not as expected. Got {}, expected around 42700".format(tasks_data_sz))
        self.assertEqual(42700, round(admin_data_sz, -2),
                         msg="admin data size was not as expected. Got {}, expected around 42700".format(admin_data_sz))
        self.assertEqual(134300, round(opps_data_sz, -2),
                         msg="Opps data size was not as expected. Got {}, expected around 134300".format(opps_data_sz))

    def test_create_model_data(self):
        """tests dataset is around 31000 observations

        Asserts: dataset length is around 32K
        """
        size = len(self.ids)
        #check that data set is accurate
        self.assertEqual(985, size,
                         msg="Model data size was not as expected. Got {}, expected around 985".format(size))

    def test_split_dataset(self):
        """tests split_dataset returns expected np array"""
        test_size = len(self.model.test_set)
        train_size = len(self.model.train_set)
        total = test_size + train_size
        self.assertEqual(0.3, round(test_size*1.0/total, -2),
                         msg="Testt data size was not as expected. Got {}, expected around 0.1".format(round(test_size*1.0/total, -1)))

    def test_create_logreg_models(self):
        num_models = len(self.model.models)
        self.assertEqual(5, num_models,
                         msg="Number of models created was not as expected. Got {}, expected around 5".format(num_models))

    def test_train_model(self):
        """test train_model method returns an ensembler that predicts"""
        self.model.train_model(self.train_features, self.train_y)
        yhat = self.model.ensembler.predict(self.train_features)
        ysize = len(yhat)
        self.assertEqual(985, ysize,
                         msg="Number of y_hat observations created was not as expected. Got {}, expected around 5".format(ysize))

    def test_eval_model(self):
        """test evaluate_model method to make sure it returns a dict"""
        self.model.train_model(self.train_features, self.train_y)
        self.model.evaluate_model(self.model.train_set, self.model.test_set)

        self.assertTrue((0 <= self.model.evaluation_metrics['test_AUC']) & (self.model.evaluation_metrics['test_AUC'] <= 1),
                        msg="Evaluation metrics test AUC not within expected range, got {}".format(self.model.evaluation_metrics['test_AUC']))
        self.assertTrue((0 <= self.model.evaluation_metrics['train_AUC']) & (self.model.evaluation_metrics['train_AUC'] <= 1),
                        msg="Evaluation metrics train AUC not within expected range, got {}".format(self.model.evaluation_metrics['train_AUC']))


if __name__ == '__main__':
    unittest.main()
