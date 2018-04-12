"""TESTS THAT THE FLASK INSTANCE IS CREATED"""
import unittest

import src.config as config
import src.main as myapp


class TestApp(unittest.TestCase):
    """TEST FLASK INSTANCE IS CREATED"""

    def test_dev_config(self):
        """ASSERTS THAT APP NAME IS APP"""
        base_config = config.BaseConfig
        flask_app = myapp.create_app(base_config)
        self.assertEqual('app', flask_app.name)


if __name__ == '__main__':
    unittest.main()
