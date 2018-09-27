"""This module holds the big query client object."""
import logging as log
import time
from google.cloud import bigquery
from google.cloud.exceptions import exceptions
from sqlalchemy import sql, text
import sqlalchemy_bigquery.base as sql_bq

from src.exceptions.api_exceptions import BadRequestError
from src.exceptions.api_exceptions import ItemNotFoundError
from src.exceptions.api_exceptions import ServiceNotAvailableError


class BigQueryClient(object):
    """Big query client object which is used to interface
    with big query projects, datasets, tables.
     The class contains a instance of the big query client
     object and uses it to connect to project, table, dataset.
     Once the connection is made the class contains methods to
     query against and insert into specified table.
     Attributes:
         project_name: string indicting the project name the
         BigQueryClient object is currently connected to.
         dataset: big query's dataset object that contains all of
         the information pretaining to the dataset the BigQueryClient
         object is connected to
         insert_record_limit: a int that represents the max insertion
         size into big query's batch insertion
     """
    _client = None
    project_name = None
    dataset = None

    def __init__(self, project_id, dataset_name, table_name,
                 service_acct_json=None):
        if service_acct_json:
            self._client = bigquery.Client.from_service_account_json(
                service_acct_json)
        else:
            self._client = bigquery.Client(project=project_id)
        self.dataset = self._client.dataset(dataset_name)
        self._table = self._client.get_table(self.dataset.table(table_name))
        self.project_name = self._client.project
        self.insert_record_limit = 500

    @property
    def table(self):
        """get the big query table value
        returns the BigQueryClient object's table object
        Returns:
            big query's table object
        """
        return self._table
    @table.setter
    def table(self, value):
        """sets the table value used for big query operations
        Args:
            value: string value which represents the name of the
            big query table. Once set, the BigQueryClient object
            will associate itself with the corresponding table object
        """
        self._table = self._client.get_table(self.dataset.table(value))

    def query(self, sql_statement):
        """runs sql statement passed in against the set big query table
        Takes the raw sql statement and runs the query against previously
        set project/dataset/table value and returns the queried rows.
        Args:
            sql_statement: raw sql statement. The sql statement expected is
            a non-legacy format for big query. The table name must be bound by
            ` character, the table name qualified as
            project_name.dataset_name.table_name
        Returns:
            A iterator object. Iterator returns bq row dictionary.
        Raises:
            ItemNotFoundError: this error is thrown by the big query client
            when api call against a non-existant project, dataset, table has
            been made
        """
        log.info('calling select statement:\n' + sql_statement)
        try:
            query_job = self._client.query(sql_statement)
            return query_job.result()
        except exceptions.NotFound as query_error:
            error_msg = ('reference not found for query statement:'
                         '{}. Error: {}'.format(sql_statement, query_error))
            self._log_raise_exception(error_msg, ItemNotFoundError)

    def insert_records(self, records):
        """inserts records passed in against set big query table
        Takes the records data and inserts it into its preconfigured
        big query project/dataset/table. If too many requests for insertions
        are made in succession gcp may return a 503 service not available.
        When this happens exponential backoff will happen where sleep time is
        given and retried. The sleep time will continue to increase until it
        reaches 32 seconds then will continue on to the next record. All
        events related to this exponential backoff will be logged.
        Args:
            records: a list of tuples object where each element in the list
            represents a row to be inserted. Each tuple represents the columns
            of a row. The tuples must be in the exact same order as the schema
            in big query table.
        Raises:
            BadRequestError: when the inputted schema of the records do not
            match, or when the inputted record is not a list of
            ServiceNotAvailableError: when exponential backoff has reached 32
            seconds but service is still unavailable.
        """
        self._validate_data(records)
        log.info('calling insert. Data: {}'.format(records))
        for _partition_records in self._partition_records(
                records, record_limit=self.insert_record_limit):
            resp = self._client.insert_rows(self.table, _partition_records)
            if len(resp) != 0:
                self._handle_insertion_error(resp, _partition_records)

    def bq_build_select_all_statement(self):
        table_name = ".".join([self.project_name,
                               self.dataset.dataset_id,
                               self.table.table_id])
        select_statement = sql.select(['*']).select_from(sql.table(table_name))\
            .compile(dialect=sql_bq.BQDialect())
        return select_statement.string.replace('[', '`').replace(']', '`')

    def _exp_backoff(self, records):
        """logic for exponential backoff. This happens when big query
        has a serve not available error. Wait time will go up to 32
        seconds and increase in time will increase by power of 2
        based on recommendation in google documentation.
        If service is not available after 32 seconds of backoff throw
        service not available error.
        """
        # google bq doc recommends waiting up to 32 seconds
        for count in range(0, 6):
            # google bq doc recommends incrementing sec by pow 2
            backoff_time = pow(2, count)
            time.sleep(backoff_time)
            resp = self._client.create_rows(self.table, records)
            if len(resp) != 0:
                self._check_schema_error(resp)
                log.error('insertion failure due to '
                          'unavailable service. '
                          'record: {} backoff time: {}'.format(records,backoff_time))
                continue
            return
        raise ServiceNotAvailableError

    @staticmethod
    def _log_raise_exception(log_msg, exception):
        """log the error message then raise the exception"""
        log.error(log_msg)
        raise exception(log_msg)

    @staticmethod
    def _partition_records(records, record_limit):
        """split the list into sub lists due to big query bulk insert limit"""
        for i in range(0, len(records), record_limit):
            yield records[i:i + record_limit]

    def _validate_data(self, records):
        """ensure records used for insertion is list of tuples"""
        if not all(isinstance(item, tuple) for item in records):
            error_msg = 'bq insert data: {} is not valid'.format(records)
            self._log_raise_exception(error_msg, BadRequestError)

    def _handle_insertion_error(self, resp, record):
        """error handling for big query insertion"""
        self._check_schema_error(resp)
        self._exp_backoff(record)

    def _check_schema_error(self, resp):
        """check for big query schema error during insertion"""
        if isinstance(resp, list):
            error_msg = 'bq insert failed due to schema: {}'.format(resp)
            log.error(error_msg)
            # if error is due to schema mismatch stop the process
            self._log_raise_exception(error_msg, BadRequestError)


