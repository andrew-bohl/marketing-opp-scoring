import unittest

import src.config as config
import src.main as app


class TestApp(unittest.TestCase):

    def test_dev_config(self):
        base_config = config.BaseConfig
        flask_app = app.create_app(base_config)
        self.assertEqual('app', flask_app.name)


if __name__ == '__main__':
    unittest.main()

