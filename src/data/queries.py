"""QUERY LOGIC FOR IMPORTING DATA"""

class QueryLogic(object):
    SALEFORCE_QUERY = """
    SELECT 
        salesforce_id,
        trial_order_detail_id, 
        trial_order_id, 
        Lead_Type__c, 
        leadsource, 
        Lead_Owner_Full_Name__c, 
        Original_Lead_Owner__c, 
        Device_Type__c, 
        Number_of_products_selling__c, 
        Reason_Not_Qualified__c, 
        SMS_Opt_in__c, 
        Timezone_editable__c, 
        TimeZone__c, 
        Industry__c, 
        SocialSignUp__c, 
        Gender__c, 
        Last_Login_from_Admin__c, 
        Have_products__c, 
        Position__c, 
        country, 
        Status, 
        lead_createdate, 
        converted_to_opp, 
        opp_createdate, 
        opp_iswon, 
        opp_CloseDate 
    FROM `v1-dev-main.LeadScoring.v1v2_leads_opps`
    WHERE lead_createdate >= '{}' and lead_createdate <= '{}'
    """

    GA_QUERY = """
    SELECT 
        demo_lookup,
        first_demo_session, 
        timestamp,
        date, 
        session_id, 
        customer_id, 
        s_version, 
        u_version, 
        landingPagePath, 
        deviceCategory, 
        adDistributionNetwork, 
        adDestinationUrl, 
        adwordsAdGroupId, 
        source, 
        medium, 
        socialNetwork, 
        demo_transaction_id, 
        u_id, 
        transaction_id, 
        campaign, 
        adgroup, 
        keyword, 
        adKeywordMatchType, 
        Creative, 
        country, 
        click_id, 
        Brand_NonBrand, 
        demo_session_position
    FROM `v1-dev-main.LeadScoring.ga_customer_paths`
    WHERE date >= '{}' and date <= '{}'
    """

    TRIAL_CONV_QUERY = """
    SELECT 
        trial_id,
        trial_date, 
        sub_id, 
        sub_date,
        CASE WHEN sub_id IS NOT NULL THEN 1 else -1 END as converted 
    FROM `v1-dev-main.LeadScoring.trial_conversions`
    WHERE trial_date >= '{}' and trial_date <= '{}'
    """

    IMPORT_SALESFORCE_LEADS = """SELECT ID, 
                            Created_Date_Time__c, 
                            order_ID__C, 
                            company
                            FROM LEAD 
                            WHERE 
                            RecordTypeID IN ('01270000000EAWZAA4', '01270000000Q4tBAAS') 
                            and CreatedDate = LAST_N_DAYS:{}"""
