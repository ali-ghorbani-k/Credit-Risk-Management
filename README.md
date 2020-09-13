# Credit-Risk-Management
Building an end-to-end machine learning model to predict the probability of paying back a loan by each applicant.

# Key Takeaways

# Data Collection
I have collected the data from kaggle that was provided by [Home Credit financial institution]( https://www.kaggle.com/c/home-credit-default-risk/data).
There are two main sources of data 1)Home Credit 2)Bureau which are presented in seven tables as:
1. __Home Credit__:

1. __application_train and application_test__: These two tables includes the information for the each loan application represented by an id of loan (__SK_ID_CURR__).
The applicatoin_train table includes a target column (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases). 

2. __previous_application__: This table includes all previous application at Home Credit which represented by an id of loan (__SK_ID_PREV__). One SK_ID_CURR can have 0,1,2 or more related previous credits in previous_application table showing a one-to-many relationship.

3. __



