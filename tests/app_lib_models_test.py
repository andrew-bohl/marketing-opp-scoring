"""test suite for data loading and model evaluation"""

from datetime import datetime as dt
import logging as log
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

        ensemble_path = self.model.config["OUTPUTS_PATH"] + 'ensemble_model/'

        if not os.path.exists(ensemble_path):
            os.makedirs(ensemble_path)

        self.start_date = dt(2018, 8, 3).date()
        self.end_date = dt(2018, 8, 14).date()

        ids_set, datasets, feat_names, sf_leadlookup = self.model.create_model_data((self.start_date, self.end_date), training=True)

        self.ids_set = ids_set
        self.datasets = datasets
        self.feat_names = feat_names
        self.sf_lookup = sf_leadlookup
        self.model.split_dataset(self.ids_set, self.datasets, test_size=0.3)

        self.model.create_logreg_models(self.model.train_set, self.feat_names)

        train_features, train_y, _, _ = self.model.create_ensemble_features(self.model.train_set, ids_set)
        test_features, test_y, _, _ = self.model.create_ensemble_features(self.model.test_set, ids_set)

        self.train_history = self.model.train_model(train_features, train_y)
        self.model.evaluate_model((test_features, test_y), (train_features, train_y))

    def test_load_data(self):
        """Tests load data function is generating approx size data"""
        data_objects = len(self.datasets)

        self.assertEqual(5, data_objects,
                         msg="Number of data objects received is not as expected. Got {}, expected around 42700".format(data_objects))

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
