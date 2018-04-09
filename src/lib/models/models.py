"""
    THIS PACKAGE BUILDS THE MODEL FROM THE CLEANED DATASET
"""

import datetime as dt
import logging as log
import pickle

import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.naive_bayes import BernoulliNB

from src.lib.bigquery import bigquery as bq
import src.lib.models.clean_data as clean_data
import src.main as app
from src.data import queries as query


class Model(object):
    """Model object class:"""
    today = dt.datetime.today().date().isoformat()

    def __init__(self):
        """instantiates model class object"""
        self.model = None
        self.features = None
        self.target = None
        self.feat_names = None
        self.evaluation_metrics = {}
        self.test_set = None
        self.train_set = None
        self.to_score = None

    def load_data(self, start_date, end_date):
        """Load data for training"""
        query_logic = query.QueryLogic
        salesforce_query = query_logic.SALEFORCE_QUERY.format(start_date, end_date)
        ga_query = query_logic.GA_QUERY.format(start_date, end_date)
        conversion_query = query_logic.TRIAL_CONV_QUERY.format(start_date, end_date)

        gcp_project_name = app.config.BQ_PROJECT_ID
        dataset_name = app.config.LEADSCORING_DATASET
        salesforce_table = app.config.SALESFORCE_TABLE

        bq_client = bq.BigQueryClient(gcp_project_name, dataset_name, salesforce_table)

        salesforce_data = clean_data.clean_salesforce_data(bq_client, salesforce_query)
        ga_paths = clean_data.clean_ga_data(bq_client, ga_query)
        trial_conversions = clean_data.clean_conversions_data(bq_client, conversion_query)

        return salesforce_data, ga_paths, trial_conversions

    def create_model_data(self, datasets, startdate, enddate):
        """Creates dataset for model training"""

        raw_data = clean_data.merge_datasets(datasets, startdate, enddate)
        features_names, target_variable, features_set, _ = clean_data.create_features(raw_data)
        self.features = features_set
        self.target = target_variable
        self.feat_names = features_names
        log.info("Model uses %d features" % (len(features_names)))

    def create_score_set(self, datasets, startdate, enddate):
        """Creates dataset for scoring"""
        score_set = clean_data.merge_datasets(datasets, startdate, enddate, False)
        _, _, features_set, id_list = clean_data.create_features(score_set, self.feat_names)

        data_dict = {}
        #id_list is pair of ids: 'trial_order_id', 'salesforce_id'
        for id_pair, feat in zip(id_list, features_set):
            temp_dict = dict()
            temp_dict['sf_id'] = id_pair[1]
            temp_dict['values'] = feat
            data_dict[id_pair[0]] = temp_dict
        self.to_score = data_dict

    def split_dataset(self, features, target, test_size=0.4, rd_state=4):
        """Split training data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=rd_state)

        shuffle_index = np.random.choice(len(X_train), len(X_train))[:]
        y_train = y_train[shuffle_index]
        X_train = X_train[shuffle_index]

        converted_index = np.where(y_train[:] == 1)
        samples_to_balance = len(converted_index[0])*2
        unconverted_index = np.where(y_train[:] == -1)
        np.random.shuffle(unconverted_index)

        new_index = np.concatenate((unconverted_index[0][:samples_to_balance], converted_index[0]))
        np.random.shuffle(new_index)

        new_y = y_train[new_index]
        new_x = X_train[new_index]

        output_path = app.config.OUTPUTS_PATH

        np.save(output_path + 'test_X', X_test)
        np.save(output_path + 'test_Y', y_test)
        np.save(output_path + 'train_X', new_x)
        np.save(output_path + 'train_Y', new_y)

        self.test_set = [y_test, X_test]
        self.train_set = [new_y, new_x]

    def train_model(self, train_set):
        """Train model for prediction task"""
        y_train = train_set[0]
        x_train = train_set[1]

        clf = BernoulliNB()
        clf.fit(x_train, y_train)
        self.model = clf
        output_path = app.config.OUTPUTS_PATH

        joblib.dump(clf, output_path + 'model_' + str(self.today) + '.pkl')

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

        output_path = app.config.OUTPUTS_PATH
        with open(output_path + 'evaluation_metrics_' + str(self.today) + '.pickle', 'wb') as handle:
            pickle.dump(self.evaluation_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

        log.info(self.evaluation_metrics)


def main():
    pass


if __name__ == '__main__':
    main()
