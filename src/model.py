"""This model contains the main function logic
which will pull data from BQ/load the data
and build model if the last model is greater than
1 week ago, else it will pull a model from GCS and
use that model to score the leads and write to Salesforce
"""
from datetime import datetime as dt, timedelta
import os

import logging as log
import pandas as pd
from google.cloud import storage

from src.lib.models import models, utilities as util
from src.lib.salesforce import salesforce


def train(start_date, end_date):
    """ Trains the model over the given date range

    :param start_date: training start date
    :param end_date: training end date
    :return: trained model
    """
    current_model = models.Model()
    output_path = current_model.config.OUTPUTS_PATH

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    salesforce_data, ga_paths, conversions = current_model.load_data(start_date, end_date)
    datasets = [salesforce_data, ga_paths, conversions]
    current_model.create_model_data(datasets, start_date, end_date)
    target = current_model.target
    features = current_model.features

    current_model.split_dataset(features, target)

    training_data = current_model.train_set
    current_model.train_model(training_data)

    testing_data = current_model.test_set
    current_model.evaluate_model(testing_data, training_data)

    bucket_name = current_model.config.BUCKET_NAME

    gcs = storage.Client()
    util.writeall_gcs_files(gcs, bucket_name, output_path)


def infer(start_date, end_date):
    """ Makes inferences over all leads created in the given date range

    :param start_date:
    :param end_date:
    :return: scored leads
    """
    #get lead data via bigquery
    current_model = models.Model()
    config = current_model.config

    gcs = storage.Client()
    bucket = util.initialize_gcs(gcs, config.BUCKET_NAME)
    model, model_name = util.load_gcs_model(bucket, config.MODEL_NAME, config.GCS_FOLDER)
    year, month, date = int(model_name[6:10]), int(model_name[11:13]), int(model_name[14:16])
    model_date = dt(year, month, date)

    if (dt.today() - model_date) <= timedelta(days=7):
        # use imported model
        current_model.model = model
        config = current_model.config

        salesforce_data, ga_paths, conversions = current_model.load_data(start_date, end_date)
        dataset = [salesforce_data, ga_paths, conversions]
        current_model.create_score_set(dataset, start_date, end_date)

        errors = []
        scores = {}
        for lead_id in current_model.to_score.keys():
            try:
                lead = model.to_score[lead_id]
                scores[lead['sf_id']] = model.model.predict_proba(lead['values'].reshape(1, -1))

            except KeyError:
                log.info(lead_id)
                log.info("Couldn't find lead in data")
                errors.append(lead_id)

        output_path = config.OUTPUTS_PATH
        pd.DataFrame(errors).to_csv(output_path+'missing_trial_ids' + str(current_model.today)+'.csv')
        pd.DataFrame.from_dict(scores, orient='index').to_csv(output_path+'sf_ids'+str(current_model.today)+'.csv')

        sf_client = salesforce.Salesforce(config)
        sf_client.write_lead_scores(scores)

        bucket_name = config.BUCKET_NAME

        util.writeall_gcs_files(gcs, bucket_name, output_path)

    else:
        log.error("Model is older than 7 days. Please train first")
        raise Exception("Model is older than 7 days")


def main():
    pass


if __name__ == '__main__':
    main()
