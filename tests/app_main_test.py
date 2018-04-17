"""TESTS THAT THE FLASK INSTANCE IS CREATED"""
import unittest

from src.config import BaseConfig
import src.main as app


class TestApp(unittest.TestCase):
    """TEST FLASK INSTANCE IS CREATED"""

    def test_dev_config(self):
        """ASSERTS THAT APP NAME IS APP"""
        base_config = BaseConfig
        flask_app = app.create_app(base_config)
        self.assertEqual('marketing-lead-scoring', flask_app.name)


if __name__ == '__main__':
    unittest.main()
