from datetime import datetime as dt, timedelta
import mock
import unittest
from google.cloud import storage

import src.config as config
import src.model as model
from src.lib.models import utilities as util

class TestModel(unittest.TestCase):
    base_config = config.BaseConfig

    def test_train(self):
        start_date = dt(2018, 2, 1)
        end_date = dt(2018, 3, 1)
        model.train(start_date, end_date)

if __name__ == '__main__':
    unittest.main()

