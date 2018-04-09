from datetime import datetime as dt, timedelta
import unittest

import src.api


class TestApp(unittest.TestCase):

    def test_format_date(self):
        date_format = "%Y-%m-%d"
        start_date_string = '2017-07-01'
        end_date_string = '2018-02-01'
        payload = {'start_date': start_date_string,
                   'end_date': end_date_string}

        date_range = src.api._format_date(payload)

        start_date_dt = dt.strptime(start_date_string, date_format)
        end_date_dt = dt.strptime(end_date_string, date_format)
        self.assertEqual((start_date_dt, end_date_dt), date_range)


if __name__ == '__main__':
    unittest.main()
