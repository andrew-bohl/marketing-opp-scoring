"""test suite for data loading and model evaluation"""

import numpy as np
import unittest

import src.config as conf
from src.lib.models import models


class ModelTests(unittest.TestCase):
    config = conf.BaseConfig

    def test_dataset_size(self):
        """
        tests dataset is around 31000 observations
        Asserts: dataset length is around 32K
        """
        test_model = models.Model()
        test_model.load_data(self.config.TEST_CUTOFF)

        #check that data set is accurate
        self.assertTrue((31000 <= len(test_model.target)) & (len(test_model.target) <= 32000),
                        msg ="Data set size not within expected range")

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
        test_model.evaluate_model(test_set)
        print(test_model.evaluation_metrics)

        train_auc = round(test_model.evaluation_metrics['train_AUC'], 2)
        test_auc = round(test_model.evaluation_metrics['test_AUC'], 2)

        self.assertEqual(0.85, train_auc,
                         msg="Train AUC not expected. Got {}, expected 0.83".format(train_auc))

        self.assertEqual(0.77, test_auc,
                         msg="Test AUC not expected. Got {}, expected 0.77".format(test_auc))


if __name__ == '__main__':
    unittest.main()
