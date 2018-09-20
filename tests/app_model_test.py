from datetime import datetime as dt, timedelta
import logging as log
import mock
import unittest

from src.main import app
import src.model as model

from src.lib.models import utilities as util


class TestModel(unittest.TestCase):
    test_data = 'tests/test_data/'

    def setUp(self):
        self.config = app.config
        log.info(self.config)

    def test_train(self):
        start_date = dt(2018, 2, 1)
        end_date = dt(2018, 3, 1)
        model.train(start_date, end_date, self.config)

if __name__ == '__main__':
    unittest.main()

