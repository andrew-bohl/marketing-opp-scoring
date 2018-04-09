"""Filter conditions for dataframe construction"""


class NotQualified(object):
    """Not Qualified Filter"""
    name = 'Reason_Not_Qualified__c'
    filters = ["Invalid Phone & Email", "Invalid Phone Number", "Internal Test"]


class LeadType(object):
    """Not Qualified Filter"""
    name = 'Lead_Type__c'
    filters = ["testmedium", "curl test"]
