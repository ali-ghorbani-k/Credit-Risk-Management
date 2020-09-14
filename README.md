# Credit-Risk-Management
Building an end-to-end machine learning model to predict the probability of paying back a loan by each applicant.

# Key Takeaways

# Data Collection
I have collected the data from kaggle that was provided by [Home Credit financial institution]( https://www.kaggle.com/c/home-credit-default-risk/data).
There are two main sources of data 1) Bureau 2) Home Credit which are presented in seven tables as:
2. __Home Credit__:

    1. __application_train__: This tables includes the information for the each loan application represented by an id of loan (__SK_ID_CURR__).
    The applicatoin_train table includes a TARGET column (1 : client with payment difficulties: he/she had late payment more than X days on at least one of the first 
    Y installments of the loan in our sample, 0 : the loan was repaid) 
    
    2. __application_test__: This table has the same column as the application_train table, but does not have TARGET column. The TARGET column will be predicted by the Machine learning model and could be used in kaggle competition.

    3. __previous_application__: This table includes all previous application at Home Credit which represented by an id of loan (__SK_ID_PREV__). One SK_ID_CURR can have 0,1,2     or more related previous credits in previous_application table showing a one-to-many relationship.
    
    4. __POS_CASH_BALANCE__: This table includes monthly balance of previous point of sale (POS) with Home Credit.
    
    5. __credit_card_balance__ : This table inlcudes monthly balance snapshots of previous credit cards that the applicant has with Home Credit
    
    6. __installments_payments__ : This table includes repayment history for the previously disbursed credits related to the loans in Home Credit database.



