"""TEST SUITE FOR SALESFORCE LIBRARY"""
import unittest

from src.main import app
from src.lib import salesforce


class SalesforceTests(unittest.TestCase):
    """UNIT TESTS MODELS PACKAGE"""
    def setUp(self):
        with app.app_context():
            self.config = app.config


if __name__ == '__main__':
    unittest.main()
