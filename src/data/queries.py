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
        CASE WHEN converted_to_opp IS TRUE then 1 else 0 END as converted_to_opp, 
        opp_createdate, 
        opp_iswon, 
        opp_CloseDate
    FROM `v1-dev-main.LeadScoring.v1v2_leads_opps`
    WHERE lead_createdate >= '{}' and lead_createdate <= '{}'
    """

    OPPS_QUERY = '''
    SELECT 
        salesforce_id,
        leadsource,
        lead_createdate,
        trial_order_detail_id,
        CASE WHEN converted_to_opp IS TRUE then 1 else 0 END as converted_to_opp, 
        opp_createdate
    FROM `v1-dev-main.LeadScoring.v1v2_leads_opps`
    WHERE lead_createdate >= '{}' and lead_createdate <= '{}'
    '''

    ADMINPV_QUERY = '''
    SELECT 
        inserted_at, 
        timestamp_millis(timestamp) as timestamp, 
        url, 
        client_id, 
        u_id, 
        sid, 
        pid, 
        property_id, 
        devicetype, 
        version, 
        lastpage, 
        landingpage, 
        optimizely, 
        continent, 
        country, 
        state, 
        city, 
        zip, 
        timezone, 
        latitude, 
        longitude 
    FROM `v1-dev-main.mrkt_data_analytics.vw_v1v2_admin_pageviews` 
    WHERE CAST(inserted_at as date) >= '{}' and CAST(inserted_at as date) <= '{}'    
    '''

    V2Clicks_QUERY = '''
    SELECT 
        inserted_at,
        tenantId, 
        action, 
        value, 
        isTrial, 
        client_id, 
        planType
    FROM `v1-dev-main.mrkt_data_analytics.trial_activity` 
    WHERE tenantid is not null and
    CAST(inserted_at as date) >= '{}' and CAST(inserted_at as date) <= '{}'
    '''

    SFTASKS_QUERY = '''
    SELECT 
        DISTINCT
        task_id, 
        lead_id, 
        opportunity_id, 
        Reason_Not_Qualified__c, 
        leadsource, 
        order_id, 
        order_detail_id,
        tenant_id__c, 
        company, 
        lead_createddate, 
        Created_Date_and_Time__c, 
        Activity_Type__c, 
        ActivityDate, 
        Call_Duration_in_Hours__c, 
        TaskSubtype, 
        WhatId, 
        Description, 
        ringdna__Call_Connected__c, 
        ringdna__Call_Duration_min__c, 
        ringdna__Queue_Hold_Time__c 
    FROM `v1-dev-main.LeadScoring.vw_tasks` 
    WHERE lead_createddate >= '{}' and lead_createddate <= '{}'    
    '''


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