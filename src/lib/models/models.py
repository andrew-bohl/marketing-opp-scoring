"""
    THIS PACKAGE BUILDS THE MODEL FROM THE CLEANED DATASET
"""

import datetime as dt
import logging as log
import math
import os
import pickle

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve
import tensorflow as tf
from tensorflow import keras

from src.lib import bigquery as bq
from src.lib.models import clean_data
from src.data import queries as query


class Model(object):
    """Model object class:"""

    def __init__(self, config):
        """instantiates model class object"""
        self.models = None
        self.ensembler = None
        self.features = None
        self.target = None
        self.evaluation_metrics = {}
        self.test_set = None
        self.train_set = None
        self.datasets = None
        self.ids = None
        self.config = config
        self.today = dt.datetime.combine(dt.datetime.today(), dt.time.min).date().isoformat()

    def load_data(self, start_date, end_date):
        """Load data for training"""
        query_logic = query.QueryLogic
        salesforce_query = query_logic.SALEFORCE_QUERY.format(start_date, end_date)
        ga_query = query_logic.GA_QUERY.format(start_date, end_date)
        opps_query = query_logic.OPPS_QUERY.format(start_date, end_date)
        admin_pv_query = query_logic.ADMINPV_QUERY.format(start_date, end_date)
        v2clicks_query = query_logic.V2Clicks_QUERY.format(start_date, end_date)
        tasks_query = query_logic.SFTASKS_QUERY.format(start_date, end_date)

        gcp_project_name = self.config["BQ_PROJECT_ID"]
        dataset_name = self.config["LEADSCORING_DATASET"]
        salesforce_table = self.config["SALESFORCE_TABLE"]

        relative_path = os.path.dirname(__file__)

        bq_client = bq.BigQueryClient(gcp_project_name, dataset_name, salesforce_table,
                                      relative_path + '/credentials/leadscoring.json')

        ga_paths = clean_data.clean_ga_data(
            bq_client,
            ga_query,
            self.config["OUTPUTS_PATH"]
        )

        salesforce_data = clean_data.clean_ga_data(
            bq_client,
            salesforce_query,
            self.config["OUTPUTS_PATH"]
        )

        opps_data = clean_data.clean_opps_data(
            bq_client,
            opps_query,
            self.config["OUTPUTS_PATH"]
        )

        admin_data = clean_data.clean_admin_data(
            bq_client,
            admin_pv_query,

            self.config["OUTPUTS_PATH"]
        )

        v2_clicks_data = clean_data.clean_v2clicks_data(
            bq_client,
            v2clicks_query,
            self.config["OUTPUTS_PATH"]
        )

        tasks_data = clean_data.clean_tasks_data(
            bq_client,
            tasks_query,
            self.config["OUTPUTS_PATH"]
        )

        return opps_data, ga_paths, tasks_data, v2_clicks_data, admin_data, salesforce_data

    def create_model_data(self, start_date, end_date, training):
        """Creates dataset for model training

        :param start_date
        :param end_date
        :param training: Boolean, true for training, false for scoring
        """
        datasets = {}
        ids = {}

        opps_data, ga_paths, tasks_data, v2_clicks_data, admin_data, salesforce_data = self.load_data(start_date, end_date)

        X_opps_sf, Y_opps_sf = clean_data.make_leads_dataset(salesforce_data)
        datasets['sf_leads'] = (X_opps_sf, Y_opps_sf)

        X_opps_a, Y_opps_a = clean_data.make_admin_dataset(admin_data, opps_data, training)
        datasets['admin'] = (X_opps_a, Y_opps_a)

        X_opps_ga, Y_opps_ga = clean_data.make_ga_dataset(ga_paths, opps_data, training)
        datasets['ga'] = (X_opps_ga, Y_opps_ga)

        X_opps_t, Y_opps_t = clean_data.make_tasks_dataset(tasks_data, opps_data, training)
        datasets['tasks'] = (X_opps_t, Y_opps_t)

        X_opps_c, Y_opps_c = clean_data.make_v2click_dataset(v2_clicks_data, opps_data, training)
        datasets['v2clicks'] = (X_opps_c, Y_opps_c)

        for tuples in datasets.values():
            for lead_id in tuples[1]:
                ids[lead_id[1]] = lead_id[0]

        if training:
            return ids, datasets
        else:
            lead_id_lookup = opps_data[['trial_order_detail_id', 'salesforce_id']].set_index(
                'trial_order_detail_id').to_dict()

            return ids, datasets, lead_id_lookup['salesforce_id']

    def split_dataset(self, ids, datasets, test_size=0.4):
        """Split training data into train and test sets"""

        def assemble_features(data, idlist):
            x_set = []
            y_set = []
            ids_set = []
            for i in range(len(data[1][:, 1])):
                if data[1][:, 1][i] in idlist:
                    x_set.append(data[0][i])
                    y_set.append(data[1][i, 0])
                    ids_set.append(data[1][i, 1])
            return x_set, y_set, ids_set

        test_ids = np.random.choice(list(ids.keys()), math.floor(len(ids.keys())*test_size), replace=False)
        train_ids = []

        for x in ids.keys():
            if x not in test_ids:
                train_ids.append(x)

        training_sets = {}
        testing_sets = {}
        for name in datasets.keys():
            a_xtrain, a_ytrain, aids_train = assemble_features(datasets[name], train_ids)
            training_sets[name] = (a_xtrain, a_ytrain, aids_train)

            a_xtest, a_ytest, aids_test = assemble_features(datasets[name], test_ids)
            testing_sets[name] = (a_xtest, a_ytest, aids_test)

        self.test_set = testing_sets
        self.train_set = training_sets

    def create_logreg_models(self, training_sets):
        """Fit individual models to each dataset"""

        models = {}
        for m_name in training_sets.keys():
            lr = LogisticRegression()
            lr.fit(np.array(training_sets[m_name][0]), np.array(training_sets[m_name][1]))
            models[m_name] = lr

        for mname in models.keys():
            joblib.dump(models[mname], 'model_' + mname + '_' + str(today) + '.pkl')

        self.models = models

    def train_model(self, feature_set, y_true, training_epoch=500):
        """trains tensorflow mode"""
        ensembler = keras.Sequential([
            keras.layers.Dense(256, activation=tf.nn.relu, input_shape=(np.array(feature_set).shape[1],)),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1)])

        optimizer = tf.train.RMSPropOptimizer(0.001)
        ensembler.compile(loss='mse', optimizer=optimizer, metrics=['mae'])

        checkpoint_path = self.config["OUTPUTS_PATH"] + "/ensemble_model/cp.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True,
                                                         period=100,
                                                         verbose=1)
        ensembler.fit(np.array(feature_set), np.array(y_true),
                      epochs=training_epoch, validation_split=0.2,
                      verbose=0, callbacks=[cp_callback])

        ensembler.save(self.config["OUTPUTS_PATH"] + '/ensemble_model/ensembleNN_model_' + str(self.today) + '.h5')
        ensembler.save_weights(self.config["OUTPUTS_PATH"] + '/ensemble_model/ensembleNN_weights_' + str(self.today) + '.h5')
        self.ensembler = ensembler

    def evaluate_model(self, test_set, train_set):
        """Generate model evaluation metrics

        :param test_set: properly formatted test feature set
        :param train_set: properly formatter train feature seut
        :param history: tensor flow training history
        :return: evaluation metrics saved a pickle
        """
        x_test, y_test = test_set[0], test_set[1]
        x_train, y_train = train_set[0], train_set[1]

        yhat_test = [1 if x > 0.5 else 0 for x in self.ensembler.predict(np.array(x_test))]
        yhat_train = [1 if x > 0.5 else 0 for x in self.ensembler.predict(np.array(x_train))]

        self.evaluation_metrics['train_confusion_matrix'] = confusion_matrix(y_train, yhat_train)
        self.evaluation_metrics['test_confusion_matrix'] = confusion_matrix(y_test, yhat_test)

        y_train_score = self.ensembler.predict(x_train)
        y_test_score = self.ensembler.predict(x_test)

        train_fpr, train_tpr, _ = roc_curve(y_train, y_train_score)
        self.evaluation_metrics['train_AUC'] = auc(train_fpr, train_tpr)

        test_fpr, test_tpr, _ = roc_curve(y_test, y_test_score)
        self.evaluation_metrics['test_AUC'] = auc(test_fpr, test_tpr)

        output_path = self.config["OUTPUTS_PATH"]
        with open(output_path + 'evaluation_metrics_opps_' + str(self.today) + '.pickle', 'wb') as handle:
            pickle.dump(self.evaluation_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

        log.info(self.evaluation_metrics)

    def write_data(self, records):
        gcp_project_name = self.config["BQ_PROJECT_ID"]
        dataset_name = self.config["LEADSCORING_DATASET"]
        table_name = self.config["OPPSCORING_TABLE"]
        relative_path = os.path.dirname(__file__)

        bq_client = bq.BigQueryClient(gcp_project_name, dataset_name, table_name,
                                      relative_path + '/credentials/leadscoring.json')

        bq_client.insert_records(records)


def create_ensemble_features(models_list, dataset, master_ids, training=True):
    """Create ensemble model inputs"""
    feature_sets = []
    for data_name in models_list.keys():
        data_dict = {}
        for i in range(len(dataset[data_name][0])):
            score = models_list[data_name].predict_proba(dataset[data_name][0][i].reshape(1, -1))
            yhat = models_list[data_name].predict(dataset[data_name][0][i].reshape(1, -1))
            y = dataset[data_name][1][i]
            data_dict[dataset[data_name][2][i]] = (score, yhat, y)
        feature_sets.append(data_dict)

    id_set = []
    for x in feature_sets:
        for k in x.keys():
            id_set.append(k)
    id_set = list(set(id_set))

    ensemble_feats = []
    y_true = []
    ids = []

    for a_id in id_set:
        x_set = []
        for feat in feature_sets.keys():
            try:
                x_set.extend(np.array([feature_sets[feat][a_id][0][0][1], feature_sets[feat][a_id][1][0]]))
            except KeyError:
                x_set.extend(np.array([0.5, 0]))
        ensemble_feats.append(np.array(x_set))
        y_true.append(master_ids[a_id])
        ids.append(a_id)
    if training:
        return ensemble_feats, y_true, ids
    else:
        return ensemble_feats, y_true, ids, feature_sets


def main():
    pass


if __name__ == '__main__':
    main()
