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

from src.data import queries
from src.lib.models import models, utilities as util
from src.lib import bigquery as bq, salesforce


def train(start_date, end_date, flask_config):
    """ Trains the model over the given date range

    :param start_date: training start date
    :param end_date: training end date
    :return: trained model
    """
    current_model = models.Model(flask_config)
    output_path = current_model.config["OUTPUTS_PATH"]

    ensemble_path = current_model.config["OUTPUTS_PATH"] + 'ensemble_model/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(ensemble_path):
        os.makedirs(ensemble_path)

    ids_set, datasets, feat_names, _ = current_model.create_model_data((start_date.date(), end_date.date()), training=True)
    current_model.split_dataset(ids_set, datasets)

    training_data = current_model.train_set
    testing_data = current_model.test_set

    current_model.create_logreg_models(training_data, feat_names)

    train_features, train_y, _, _ = current_model.create_ensemble_features(training_data, ids_set)
    test_features, test_y, _, _ = current_model.create_ensemble_features(testing_data, ids_set)

    train_history = current_model.train_model(train_features, train_y)
    current_model.evaluate_model((test_features, test_y), (train_features, train_y))

    bucket_name = current_model.config["BUCKET_NAME"]
    gcs = storage.Client.from_service_account_json('src/credentials/leadscoring.json')
    util.writeall_gcs_files(gcs, bucket_name, output_path)
    util.writeall_gcs_files(gcs, bucket_name, output_path+'ensemble_model/')


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
        date_range = (start_date, end_date)
        features = {}
        for model_name in logreg_models.keys():
            features[model_name] = logreg_models[model_name].features_names

        ids_set, datasets, features, sf_leadlookup = current_model.create_model_data(date_range, features, training=False)
        current_model.split_dataset(ids_set, datasets, test_size=0.0)
        new_dataset = current_model.train_set
        ensemble_features, opp_values, trial_ids, feature_sets = current_model.create_ensemble_features(new_dataset, ids_set)

        opp_values = np.array([1 if y else 0 for y in opp_values])
        ensemble_features = np.array(ensemble_features)
        trial_ids = np.array(trial_ids)
        ix_list = np.arange(len(opp_values))
        score_ix = ix_list[np.where(opp_values != 1)]

        ensembler_scores = current_model.ensembler.predict(ensemble_features)
        good, best = util.get_percentile_groups(ensembler_scores)

        records = []
        for i in score_ix:
            insert_at = dt.now()
            trial_order_detail_id = trial_ids[i]
            try:
                lead_id = sf_leadlookup[trial_order_detail_id]
            except KeyError:
                log.info("No lead_id found for trial: %s" % trial_order_detail_id)
                lead_id = None
            NN_score = ensembler_scores[i][0].astype('float64')
            sf_leads_score = ensemble_features[i][0].astype('float64')
            sf_leads_yhat = ensemble_features[i][1]
            admin_score = ensemble_features[i][2].astype('float64')
            admin_yhat = ensemble_features[i][3]
            ga_score = ensemble_features[i][4].astype('float64')
            ga_yhat = ensemble_features[i][5]
            tasks_score = ensemble_features[i][6].astype('float64')
            tasks_yhat = ensemble_features[i][7]
            v2clicks_score = ensemble_features[i][8].astype('float64')
            v2clicks_yhat = ensemble_features[i][9]
            if NN_score >= good:
                if NN_score < best:
                    label = 'best' #we are switching the names as per sales request
                else:
                    label = 'better' #good will actually be best in order to get sales to call more
            else:
                label = 'good'

            a_record = [insert_at, lead_id, trial_order_detail_id, NN_score,
                        ga_score, ga_yhat,
                        tasks_score, tasks_yhat,
                        admin_score, admin_yhat,
                        sf_leads_score, sf_leads_yhat,
                        v2clicks_score, v2clicks_yhat,
                        label]
            records.append(tuple(a_record))
            # current_model.write_data([tuple(a_record)])

        current_model.write_data(records)

    else:
        log.error("Model is older than 30 days. Please train first")
        log.info("Model is older than 30 days.. Retraining")
        midnight = dt.combine(dt.today(), time.min)
        start_date = midnight - timedelta(days=180)
        end_date = midnight - timedelta(days=30)
        train(start_date, end_date, flask_config)
        infer(start_date, end_date, flask_config)


def write_scores(startdate, enddate, flask_config):
    bq_client = bq.BigQueryClient(flask_config["BQ_PROJECT_ID"], flask_config["LEADSCORING_DATASET"],
                                  flask_config["OPPSCORING_TABLE"],
                                  'src/credentials/leadscoring.json')
    scores_query = queries.QueryLogic.NN_SCORES_QUERY
    scores_dataframe = util.load_bigquery_data(bq_client, scores_query)
    scores = scores_dataframe.set_index('lead_id').to_dict()
    sf_client = salesforce.salesforce_api(flask_config, sandbox=False)
    sf_client.write_lead_scores(scores['label'])

