"""This model contains the main function logic
which will pull data from BQ/load the data
and build model if the last model is greater than
1 week ago, else it will pull a model from GCS and
use that model to score the leads and write to Salesforce
"""
from datetime import datetime as dt, time, timedelta
import os

import logging as log
import numpy as np
from google.cloud import storage

from src.lib.models import models, utilities as util


def train(start_date, end_date, flask_config):
    """ Trains the model over the given date range

    :param start_date: training start date
    :param end_date: training end date
    :return: trained model
    """
    current_model = models.Model(flask_config)
    output_path = current_model.config["OUTPUTS_PATH"]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ids_set, datasets = current_model.create_model_data(start_date.date(), end_date.date(), True)
    current_model.split_dataset(ids_set, datasets)

    training_data = current_model.train_set
    testing_data = current_model.test_set

    current_model.create_logreg_models(training_data)
    model_list = current_model.models

    train_features, train_y, _ = current_model.create_ensemble_features(model_list, training_data, ids_set)
    test_features, test_y, _ = current_model.create_ensemble_features(model_list, testing_data, ids_set)

    train_history = current_model.train_model(train_features, train_y)
    current_model.evaluate_model((test_features, test_y), (train_features, train_y), train_history)

    testing_data = current_model.test_set
    current_model.evaluate_model(testing_data, training_data)

    bucket_name = current_model.config["BUCKET_NAME"]
    gcs = storage.Client.from_service_account_json('src/credentials/leadscoring.json')
    util.writeall_gcs_files(gcs, bucket_name, output_path)


def infer(start_date, end_date, flask_config):
    """ Makes inferences over all leads created in the given date range
    :param start_date:
    :param end_date:
    :return: scored leads
    """
    current_model = models.Model(flask_config)
    config = current_model.config
    output_path = config["OUTPUTS_PATH"]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gcs = storage.Client.from_service_account_json('src/credentials/leadscoring.json')

    bucket = util.initialize_gcs(gcs, config["BUCKET_NAME"])
    logreg_models, ensembler, model_date = util.load_gcs_model(bucket, config["MODEL_NAME"], config["GCS_FOLDER"])

    year, month, date = int(model_date[:4]), int(model_date[4:6]), int(model_date[6:])
    model_date = dt(year, month, date)

    if (dt.today() - model_date) <= timedelta(days=30):
        # use imported model
        current_model.models = logreg_models
        current_model.ensembler = ensembler
        ids_set, datasets, lead_lookup = current_model.create_model_data(start_date.date(), end_date.date(), False)
        features, opp_values, trial_ids, feature_sets = models.create_ensemble_features(logreg_models, datasets,
                                                                                        ids_set, training=False)

        opp_values = np.array([1 if y else 0 for y in opp_values])
        features = np.array(features)
        trial_ids = np.array(trial_ids)
        score_ix = np.where(opp_values == 0)

        ensembler_scores = current_model.ensembler.predict(features)
        good, best = util.get_percentile_groups(ensembler_scores)

        records = []
        for i in score_ix:
            insert_at = dt.now()
            trial_order_detail_id = trial_ids[i]
            try:
                lead_id = lead_lookup[trial_order_detail_id]
            except KeyError:
                log.info("No lead_id found for trial: %s" % trial_order_detail_id)
                lead_id = None
            ensembler_score = ensembler_scores[i]
            ga_score = feature_sets['ga'][trial_order_detail_id][0][0][1]
            ga_yhat = feature_sets['ga'][trial_order_detail_id][1][0]
            tasks_score = feature_sets['tasks'][trial_order_detail_id][0][0][1]
            tasks_yhat = feature_sets['tasks'][trial_order_detail_id][1][0]
            admin_score = feature_sets['admin'][trial_order_detail_id][0][0][1]
            admin_yhat = feature_sets['admin'][trial_order_detail_id][1][0]
            sf_leads_score = feature_sets['sf_leads'][trial_order_detail_id][0][0][1]
            sf_leads_yhat = feature_sets['sf_leads'][trial_order_detail_id][1][0]
            v2clicks_score = feature_sets['v2clicks'][trial_order_detail_id][0][0][1]
            v2clicks_yhat = feature_sets['v2clicks'][trial_order_detail_id][1][0]
            if ensembler_score > good:
                if ensembler_score < best:
                    label = 'better'
                else:
                    label = 'best'
            else:
                label = 'good'
            a_record = (insert_at, lead_id, trial_order_detail_id, ensembler_score,
                        ga_score, ga_yhat,
                        tasks_score, tasks_yhat,
                        admin_score, admin_yhat,
                        sf_leads_score, sf_leads_yhat,
                        v2clicks_score, v2clicks_yhat,
                        label)
            records.append(a_record)

        current_model.write_data(records)
        util.writeall_gcs_files(gcs, config["BUCKET_NAME"], output_path)

    else:
        log.error("Model is older than 30 days. Please train first")
        log.info("Model is older than 30 days.. Retraining")
        midnight = dt.combine(dt.today(), time.min)
        start_date = midnight - timedelta(days=180)
        end_date = midnight - timedelta(days=30)
        train(start_date, end_date, flask_config)
        infer(start_date, end_date, flask_config)
