"""TEST SUITE FOR SALESFORCE LIBRARY"""
import unittest

import src.config as conf
from src.lib.salesforce import salesforce


class SalesforceTests(unittest.TestCase):
    """UNIT TESTS MODELS PACKAGE"""
    config = conf.BaseConfig


if __name__ == '__main__':
    unittest.main()
