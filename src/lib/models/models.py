"""
    THIS PACKAGE BUILDS THE MODEL FROM THE CLEANED DATASET
"""
import pickle

import datetime as dt
import numpy as np

from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.naive_bayes import BernoulliNB

import src.config as config
from src.lib.bigquery import bigquery as bq
import src.lib.models.clean_data as clean_data
from src.data import queries as qry



class Model(object):
    """Model object class:"""
    def __init__(self):
        """
        :param self: model instance
        :param sklearn_model: takes a sklearn mode
        :return: instantiates model object
        """
        self.model = None
        self.features = None
        self.target = None
        self.feat_names = None
        self.evaluation_metrics = {}
        self.test_set = None
        self.train_set = None

    def load_data(self, cutoff = dt.datetime(2018, 2, 1)):
        """Load data for training"""
        query_logic = qry.QueryLogic
        gcp_project_name = config.BaseConfig.BQ_PROJECT_ID
        dataset_name = config.BaseConfig.LEADSCORING_DATASET
        salesforce_table = config.BaseConfig.SALESFORCE_TABLE

        bq_client = bq.BigQueryClient(gcp_project_name, dataset_name, salesforce_table)

        salesforce_data = clean_data.clean_salesforce_data(bq_client, query_logic.SALEFORCE_QUERY)
        ga_paths = clean_data.clean_ga_data(bq_client, query_logic.GA_QUERY)
        trial_conversions = clean_data.clean_conversions_data(bq_client, query_logic.TRIAL_CONV_QUERY)

        raw_data = clean_data.merge_datasets(salesforce_data, ga_paths, trial_conversions, cutoff)
        features_names, target_variable, features_set = clean_data.create_features(raw_data)
        self.features = features_set
        self.target = target_variable
        self.feat_names = features_names


    def split_dataset(self,test_sz=0.4, rd_state=4):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size=test_sz, random_state=rd_state)

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

        np.save('test_X', X_test)
        np.save('test_Y', y_test)
        np.save('train_X', new_x)
        np.save('train_Y', new_y)

        self.test_set = [y_test, X_test]
        self.train_set = [new_y, new_x]


    def train_model(self, train_set):
        """Train model for prediction task
        TODO: add in date parameter
        """

        y_train = train_set[0]
        x_train = train_set[1]

        clf = BernoulliNB()
        clf.fit(x_train, y_train)
        self.model = clf

        today = dt.datetime.now().date().isoformat()
        joblib.dump(clf, 'model_'+str(today)+'.pkl')


    def evaluate_model(self, test_data):
        """
        Generate model evaluation emtrics
        :return: self.evaluation_metrics, a dict
        """

        self.test_set = test_data

        def classify(classifier, data):
            """
            helper function to lower prob thresholds for labeling
            :param classifier:
            :param data:
            :return:
            """
            probs = classifier.predict_proba(data)
            labels = []
            for i in probs:
                if i[1] > 0.20:
                    labels.append(1)
                else:
                    labels.append(-1)
            return labels

        y_test, x_test = self.test_set[0], self.test_set[1]
        y_train, x_train = self.train_set[0], self.train_set[1]

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


def main():
    """Main builds and runs the model"""
    new_model = Model()

    #will need to run this after all the load marketiing report jobs in xplenty
    new_model.load_data()
    new_model.split_dataset()

    train_set = new_model.train_set
    test_set = new_model.test_set

    new_model.train_model(train_set, test_set)
    new_model.evaluate_model(test_set)
    print(new_model.evaluation_metrics)


if __name__ == '__main__':
    main()
