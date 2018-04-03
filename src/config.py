""" Base Application Configuration

    The class in this modules stores all of the various
    configuration information. The class itself does not
    contain any methods it only contains constants that is
    referred by other modules
"""
import datetime as dt
import os


class BaseConfig(object):
    """Class that contains the base config"""
    APP_NAME = "app"
    ENV = os.getenv('ENV', 'dev')

    # big query configurations
    BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", 'v1-dev-main')
    LEADSCORING_DATASET = os.getenv("LEADSCORING_DATASET", 'LeadScoring')
    SALESFORCE_TABLE = os.getenv("SALESFORCE_TABLE", 'v1v2_leads_opps')
    TRIAL_CONVERSION_TABLE = os.getenv("TRIAL_CONVERSION_TABLE", 'trial_conversions')
    GA_SESSIONS = os.getenv("GA_SESSIONS_TABLE", 'ga_customer_paths')

    BUCKET_NAME = 'marketing-lead-scoring'
    GCS_FOLDER = ''

    """ TEST CONFIG E.G CUT OFF DATES FOR TESTING """
    TEST_CUTOFF = dt.datetime(2018, 2, 1)
