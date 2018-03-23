""" Base Application Configuration

    The class in this modules stores all of the various
    configuration information. The class itself does not
    contain any methods it only contains constants that is
    referred by other modules
"""
import os


class BaseConfig(object):
    """Class that contains the base config"""
    APP_NAME = "app"
    ENV = os.getenv('ENV', 'dev')

    # big query configurations
    # BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", 'v1-dev-main')
    # IR_DATA_SET = os.getenv("IR_DATA_SET", 'affiliates')
    # STORE_CONVERSION_TABLE = os.getenv(
    #     "STORE_CONVERSION_TABLE", 'vw_IR_unified_SC_test')
    # FREE_TRIAL_CONVERSION_TABLE = os.getenv(
    #     "FREE_TRIAL_CONVERSION_TABLE", 'vw_IR_unified_FTC_test')
    # IR_LOG_TABLE = os.getenv(
    #     "IR_LOG_TABLE", 'ir_conversion_log_test')
    # BQ_INSERT_BATCH_SIZE = 500

    # # IR configurations
    # IR_URL = 'https://api.impactradius.com'
    # IMPACT_RADIUS_ACCOUNT_SID = os.getenv('IMPACT_RADIUS_ACCOUNT_SID')
    # IMPACT_RADIUS_AUTH_TOKEN = os.getenv('IMPACT_RADIUS_AUTH_TOKEN')
    # CAMPAIGN_ID = os.getenv('CAMPAIGN_ID')
    # STORE_ACTION_TRACKER_ID = os.getenv('STORE_ACTION_TRACKER_ID')
    # FREE_TRIAL_ACTION_TRACKER_ID = os.getenv('FREE_TRIAL_ACTION_TRACKER_ID')
    # AFFILIATES = [{'table_name': FREE_TRIAL_CONVERSION_TABLE,
    #                'tracker_id': FREE_TRIAL_ACTION_TRACKER_ID},
    #               {'table_name': STORE_CONVERSION_TABLE,
    #                'tracker_id': STORE_ACTION_TRACKER_ID}]

