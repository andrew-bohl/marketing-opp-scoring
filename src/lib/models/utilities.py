""" Module helper functions"""
import logging as log
import os

import numpy as np
import pandas as pd
from sklearn.externals import joblib


def load_bigquery_data(bq_client, sql):
    """
    :param bq_client: bigquery client
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


def download_blob(bucket, filename, foldername=None):
    if foldername:
        blob = bucket.blob(foldername + "/" + filename)
    else:
        blob = bucket.blob(filename)
    blob.download_to_filename(filename)


def initialize_gcs(gcs_client, bucketname):
    """initializes the gcs bucket
    :param gcs_client: gcs client
    :param bucketname: name of bucket
    """
    bucket = gcs_client.get_bucket(bucketname)
    return bucket


def load_gcs_data(bucket, filename, foldername=None):
    """ Loads datafiles for data creation and returns a pandas dataframe

    :param bucket: gcs client bucket
    :param filename: filename to load
    :param foldername; folder location of file
    :return: dataframe
    """
    download_blob(bucket, filename, foldername)
    data = pd.read_csv(filename, index_col=False, low_memory=False, dtype=str)
    return data


def write_gcs_file(bucket, filename, dest_filename):
    """Writes file to gcs bucket

    :param bucket: bucket to write to
    :param filename: name of file currently in local
    :param dest_filename: name of file at destination
    :return: success message
    """
    blob = bucket.blob(dest_filename)
    blob.upload_from_filename(filename)

    log.info('File {} uploaded to bucket.'.format(filename))


def writeall_gcs_files(gcs_client, bucket_name, filepath='.'):
    """Writes all files in local path to gcs

    :param gcs_client: gcs client
    :param bucket_name: name of gcs bucket
    :param filepath: local folder path to check for
    """
    bucket = gcs_client.get_bucket(bucket_name)
    filenames = [f for f in os.listdir(filepath) if os.path.isfile(filepath + f)]
    for file in filenames:
        print(file)
        write_gcs_file(bucket, os.path.join(filepath, file), file)


def load_gcs_model(bucket, model_name=None, foldername=None):
    """ Loads datafiles for data creation and returns a pandas dataframe

    :param bucket: gcs bucket
    :param model_name: name of model
    :param foldername; folder location of file
    :return: dataframe
    """
    if model_name:
        download_blob(bucket, model_name, foldername)
    # return the latest model
    else:
        model_names = []
        model_dates = []
        model_features = []
        for blob in bucket.list_blobs(prefix='opps_model_'):
            model_names.append(blob.name)
            model_dates.append(int(blob.name[11:-4].replace('-', '')))
            model_features.append('opp_features_names_' + blob.name[11:-4] +'.npy')

        model_name = model_names[np.argmax(model_dates)]
        feature_names = model_features[np.argmax(model_dates)]
        download_blob(bucket, model_name, foldername)
        download_blob(bucket, feature_names, foldername)

    feature_names = np.load(feature_names)
    model = joblib.load(model_name)
    return model, model_name, feature_names

def load_gcs_scores(bucket, score_file=None, foldername=None):
    """ Loads datafiles for data creation and returns a pandas dataframe

    :param bucket: gcs bucket
    :param score_file: name of scores file
    :param foldername; folder location of file
    :return: dataframe
    """
    if score_file:
        download_blob(bucket, score_file, foldername)
    # return the latest model
    else:
        file_names = []
        file_dates = []
        for blob in bucket.list_blobs(prefix='sf_ids'):
            file_names.append(blob.name)
            file_dates.append(int(blob.name[6:-4].replace('-', '')))

        try:
            file_name = file_names[np.argmax(file_dates)]
            download_blob(bucket, file_name, foldername)
        except ValueError:
            return None

    scores = pd.read_csv(file_name)
    scores.set_index("Unnamed: 0").to_dict(orient='index')
    return scores


def standardize_columns(dataframe):
    dataframe.columns = [x.lower() for x in dataframe.columns]
    return dataframe


def filter_data(filter, DataFrame):
    """filters dataframe"""
    data = DataFrame[~(DataFrame[filter.name].isin(filter.filters))]
    return data


def convert_cols_to_datetime(dataframe):
    """converts date columns to datetime

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

