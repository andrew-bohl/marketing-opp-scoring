""" Salesforce module for pulling lead ids and writing the scored values """
import time

import logging as log
import pandas as pd
import numpy as np
from simple_salesforce import Salesforce
from simple_salesforce.exceptions import SalesforceResourceNotFound, SalesforceMalformedRequest

import src.data.queries as query


class salesforce_api(object):
    """Salesforce object for instantiating client"""

    def __init__(self, config, sandbox=True):

        # instantiate simple salesforce
        if sandbox:
            self.sf_client = Salesforce(username=config["SALESFORCE_USERNAME"],
                                        password=config["SALESFORCE_PASSWORD"],
                                        security_token=config["SALESFORCE_TOKEN"],
                                        sandbox=True)
        else:
            self.sf_client = Salesforce(username=config["SALESFORCE_USERNAME"],
                                        password=config["SALESFORCE_PASSWORD"],
                                        security_token=config["SALESFORCE_TOKEN"])
        self.records = None
        self.data = None

    def get_salesforce_leads(self, num_days):
        """ pull last N days of SF leads

        :param num_days: last N days of leads to pull
        :return: dataframe with leads
        """
        query_logic = query.QueryLogic.IMPORT_SALESFORCE_LEADS.format(str(num_days))
        sf_query = self.sf_client.query_all(query_logic)
        self.records = sf_query['records']
        total_size = sf_query['totalSize']

        while not sf_query['done'] and len(self.records) < total_size:
            next_records_url = sf_query['nextRecordsUrl']
            query_result = self.sf_client.query_more(next_records_url)
            self.records = self.records + query_result['records']

    def create_salesforce_keys(self):
        """ Creates a lookup between SF ids and trial order ids"""
        salesforce_df = pd.DataFrame.from_dict(self.records, orient='columns')
        salesforce_df['Order_ID__c'] = salesforce_df['Order_ID__c'].combine_first(salesforce_df['Company'])
        self.data = salesforce_df[['Id', 'Order_ID__c', 'Created_Date_Time__c']]

    def write_lead_scores(self, scores, old_scores=None):
        """ Writes lead scorees to salesforce

        :param scores: Dict of scores where the order_id is the key and value is score
        :return: updates leads in sf
        """
        for lead in scores:
            sf_id = lead
            score = round(scores[lead][1]*100, -1)

            a_num = np.random.choice(100)
            if a_num > 90:
                time.sleep(3)

            try:
                prev_score = round(old_scores[sf_id][1]*100, -1)
            except (KeyError, TypeError):
                prev_score = 0

            if score > prev_score:
                try:
                    log.info("SF LEAD:{0} has score {1}".format(lead, score))
                    self.sf_client.Lead.update(sf_id, {'Comm_score__c': score}, headers={"Sforce-Auto-Assign": "False"})
                except SalesforceResourceNotFound as e:
                    log.error(e.content)
                except SalesforceMalformedRequest as e:
                    log.error(e.content)
                except TypeError:
                    log.error("TypeError for lead: {0} with score {1}".format(lead, score))



