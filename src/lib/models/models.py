"""
    THIS PACKAGE BUILDS THE MODEL FROM THE CLEANED DATASET
"""

import datetime as dt
import logging as log
import os
import pickle

import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.svm import SVC

from src.lib import bigquery as bq
from src.lib.models import clean_data
from src.data import queries as query


class Model(object):
    """Model object class:"""
    today = dt.datetime.today().date().isoformat()

    def __init__(self, config):
        """instantiates model class object"""
        self.model = None
        self.features = None
        self.target = None
        self.feat_names = None
        self.evaluation_metrics = {}
        self.test_set = None
        self.train_set = None
        self.to_score = None
        self.config = config

    def load_data(self, start_date, end_date):
        """Load data for training"""
        query_logic = query.QueryLogic
        salesforce_query = query_logic.SALEFORCE_QUERY.format(start_date, end_date)
        ga_query = query_logic.GA_QUERY.format(start_date, end_date)

        gcp_project_name = self.config["BQ_PROJECT_ID"]
        dataset_name = self.config["LEADSCORING_DATASET"]
        salesforce_table = self.config["SALESFORCE_TABLE"]

        relative_path = os.path.dirname(__file__)

        bq_client = bq.BigQueryClient(gcp_project_name, dataset_name, salesforce_table,
                                      relative_path + '/credentials/leadscoring.json')

        salesforce_data = clean_data.clean_salesforce_data(
            bq_client, 
            salesforce_query, 
            self.config["OUTPUTS_PATH"]
        )
        ga_paths = clean_data.clean_ga_data(
            bq_client, 
            ga_query, 
            self.config["OUTPUTS_PATH"]
        )

        return salesforce_data, ga_paths

    def create_model_data(self, datasets, startdate, enddate):
        """Creates dataset for model training"""

        raw_data = clean_data.merge_datasets(
            datasets,
            startdate, 
            enddate,
            self.config["OUTPUTS_PATH"],
        )
        features_names, target_variable, features_set, _ = clean_data.create_features(
            raw_data,
            self.config["OUTPUTS_PATH"]
        )
        pca_X = clean_data.pca_transform(
            features_set,
            features_names,
            50,
            self.config["OUTPUTS_PATH"]
        )

        self.features = pca_X
        self.target = target_variable
        self.feat_names = features_names
        log.info("Model uses %d features" % (len(features_names)))

    def create_score_set(self, datasets, startdate, enddate):
        """Creates dataset for scoring"""
        score_set = clean_data.merge_datasets(
            datasets, 
            startdate, 
            enddate, 
            self.config["OUTPUTS_PATH"]
        )
        _, _, features_set, id_list = clean_data.create_features(
            score_set, 
            self.config["OUTPUTS_PATH"],
            self.feat_names
        )
        pca_X = clean_data.pca_transform(
            features_set,
            features_names,
            50,
            self.config["OUTPUTS_PATH"]
        )
        data_dict = {}
        #id_list is pair of ids: 'trial_order_id', 'salesforce_id'
        for id_pair, feat in zip(id_list, pca_X):
            temp_dict = dict()
            temp_dict['sf_id'] = id_pair[1]
            temp_dict['values'] = feat
            data_dict[id_pair[0]] = temp_dict
        self.to_score = data_dict

    def split_dataset(self, features, target, test_size=0.4, rd_state=8):
        """Split training data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=rd_state)

        output_path = self.config["OUTPUTS_PATH"]

        np.save(output_path + 'test_X', X_test)
        np.save(output_path + 'test_Y', y_test)
        np.save(output_path + 'train_X', X_train)
        np.save(output_path + 'train_Y', y_train)

        self.test_set = [y_test, X_test]
        self.train_set = [y_train, X_train]

    def train_model(self, train_set):
        """Train model for prediction task"""
        y_train = train_set[0]
        x_train = train_set[1]

        clf = SVC(probability=True)
        clf.fit(x_train, y_train)
        self.model = clf
        output_path = self.config["OUTPUTS_PATH"]

        joblib.dump(clf, output_path + 'opps_model_' + str(self.today) + '.pkl')

    def evaluate_model(self, test_set, train_set):
        """Generate model evaluation metrics"""
        def classify(classifier, data):
            """helper function to lower prob thresholds for labeling"""
            probs = classifier.predict_proba(data)
            labels = []
            for i in probs:
                if i[1] > 0.30:
                    labels.append(1)
                else:
                    labels.append(-1)
            return labels

        y_test, x_test = test_set[0], test_set[1]
        y_train, x_train = train_set[0], train_set[1]

        train_cv_score = cross_val_score(self.model, x_train, y_train, cv=10)
        self.evaluation_metrics['cv_train_accuracy'] = train_cv_score.mean()

        test_cv_score = cross_val_score(self.model, x_test, y_test, cv=10)
        self.evaluation_metrics['cv_test_accuracy'] = test_cv_score.mean()

        yhat_test = classify(self.model, x_test)
        yhat_train = classify(self.model, x_train)
        self.evaluation_metrics['threshold_test_accuracy'] = accuracy_score(y_test, yhat_test)
        self.evaluation_metrics['threshold_train_accuracy'] = accuracy_score(y_train, yhat_train)

        self.evaluation_metrics['train_confusion_matrix'] = confusion_matrix(y_train, yhat_train)
        self.evaluation_metrics['test_confusion_matrix'] = confusion_matrix(y_test, yhat_test)

        y_train_score = self.model.predict_proba(x_train)
        y_test_score = self.model.predict_proba(x_test)

        train_fpr, train_tpr, _ = roc_curve(y_train, y_train_score[:, 1])
        self.evaluation_metrics['train_AUC'] = auc(train_fpr, train_tpr)

        test_fpr, test_tpr, _ = roc_curve(y_test, y_test_score[:, 1])
        self.evaluation_metrics['test_AUC'] = auc(test_fpr, test_tpr)

        output_path = self.config["OUTPUTS_PATH"]
        with open(output_path + 'evaluation_metrics_opps_' + str(self.today) + '.pickle', 'wb') as handle:
            pickle.dump(self.evaluation_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

        log.info(self.evaluation_metrics)


def main():
    pass


if __name__ == '__main__':
    main()
