""" Module uses """
import os

import pandas as pd


def load_bigquery_data(bq_client, sql):
    """
    :param client: bigquery client
    :param sql: sql query
    :return: dataframe
    """
    payload = bq_client.query(sql)
    cols = [x._name for x in payload.schema]

    data_array = [row.values() for row in payload]
    dataframe = pd.DataFrame(data_array, columns=cols)

    def convert_ids_to_str(a_df):
        for col in a_df.columns:
            if col.find('_id') >= 0:
                a_df[col] = a_df[col].apply(lambda x: str(x))
        return a_df

    dataframe = convert_ids_to_str(dataframe)
    return dataframe


def load_gcs_data(gcs_client, bucket_name, filename, foldername=None):
    """ Loads datafiles for data creation and returns a pandas dataframe
    :param gcs: gcs client
    :param filename: filename to load
    :param foldername; folder location of file
    :return: dataframe
    """
    bucket = gcs_client.get_bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.download_to_filename(foldername + "/" + filename)
    data = pd.read_csv(filename, index_col=False, low_memory=False, dtype=str)
    return data

def standardize_columns(dataframe):
    dataframe.columns = [x.lower() for x in dataframe.columns]
    return dataframe


def filter_data(filter, DataFrame):
    """
    :param filter: filter class
    :param DataFrame: dataframe to filter
    :return: dataframe
    """
    data = DataFrame[~(DataFrame[filter.name].isin(filter.filters))]
    return data


def convert_cols_to_datetime(dataframe):
    """
    :param dataframe: dataframe
    :return: dataframe with date columns converted to datetime
    """
    for col in dataframe.columns:
        if col.endswith('date'):
            dataframe[col] = pd.to_datetime(dataframe[col])
    return dataframe


def main():
    pass


if __name__ == '__main__':
    main()


