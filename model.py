import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from functools import reduce
import gc
import featuretools as ft
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, cohen_kappa_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.cluster import AgglomerativeClustering
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
from pdb import set_trace as st


class classifier:

    def __init__(self, args):
        self.ft_maxdep = args.ft_maxdep
        self.cnn_bsize = args.cnn_bsize
        self.cnn_epoch = args.cnn_epoch
        self.rnn_bsize = args.rnn_bsize
        self.rnn_epoch = args.rnn_epoch
        self.pca_n = args.pca_n
        self.nfolds = args.nfolds
        self.test_size = args.test_size
        self.batch_size = args.batch_size
        self.epoch = args.epoch

        if args.use_ftools:
            self.data = self.automated_features
        elif args.use_cnnft:
            self.data = self.bureau_cnn_features()
        elif args.use_rnnft:
            self.data = self.bureau_rnn_features()
        else:
            self.data = self.merge_tables()

    def hc_prv_app(self):
        """
        This method reads previous_application table that includes the recorded
        previous credits at Home Credit financial institution, perform manually feature engineering,
        flatten multiple loans, and returns the statistics related to each application SK_ID_CURR.
        """

        print('Processing previous_application table related to Home Credit source...')
        prev = pd.read_csv('data/previous_application.csv')

        # ------------------------------Feature Engineering (1): General ---------------------------
        # When was the last application applied and contract status?
        prev1 = prev.sort_values('DAYS_DECISION', ascending=False). \
            groupby(['SK_ID_CURR']).agg(
            {'DAYS_DECISION': 'first', 'NAME_CONTRACT_STATUS': 'first', 'AMT_CREDIT': 'first'})
        # last credit amount, interest rates, ... the most recent approved
        df = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved'] \
            .sort_values('DAYS_DECISION', ascending=False).groupby('SK_ID_CURR').first()
        df = df[['NAME_CONTRACT_TYPE', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT',
                 'NAME_YIELD_GROUP', 'NFLAG_INSURED_ON_APPROVAL', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED']]
        df['Diff_applied_apprved'] = df['AMT_APPLICATION'] - df['AMT_CREDIT']
        prev1 = prev1.merge(df, on='SK_ID_CURR', how='outer')

        # ----Feature Engineering (2): Ratios of Approved, Refused, Canceled, and Unused offer for each application-----
        df = prev.groupby(['SK_ID_CURR', 'NAME_CONTRACT_STATUS']).agg({'SK_ID_PREV': 'count'})
        df = df.groupby(level='SK_ID_CURR').apply(lambda x: x / x.sum())
        approved = df[df.index.isin(['Approved'], level='NAME_CONTRACT_STATUS')]['SK_ID_PREV']
        approved.index = approved.index.droplevel('NAME_CONTRACT_STATUS')
        refused = df[df.index.isin(['Refused'], level='NAME_CONTRACT_STATUS')]['SK_ID_PREV']
        refused.index = refused.index.droplevel('NAME_CONTRACT_STATUS')
        canceled = df[df.index.isin(['Canceled'], level='NAME_CONTRACT_STATUS')]['SK_ID_PREV']
        canceled.index = canceled.index.droplevel('NAME_CONTRACT_STATUS')
        unused = df[df.index.isin(['Unused offer'], level='NAME_CONTRACT_STATUS')]['SK_ID_PREV']
        unused.index = unused.index.droplevel('NAME_CONTRACT_STATUS')
        data_frames = [approved, refused, canceled, unused]
        df = reduce(lambda left, right: pd.merge(left, right, on='SK_ID_CURR', how='outer'), data_frames)
        df.columns = ['ratio_approved', 'ratio_refused', 'ratio_canceled', 'ratio_unused']
        prev1 = prev1.merge(df, on='SK_ID_CURR', how='outer')

        # -----Feature Engineering (3): Latest credit time and amount for the approved different NAME_CONTRACT_TYPE ----
        df = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved'].sort_values('DAYS_DECISION', ascending=False). \
            groupby(['SK_ID_CURR', 'NAME_CONTRACT_TYPE']).agg({'DAYS_DECISION': 'first', 'AMT_CREDIT': 'first'})
        consumer_loans = df[df.index.isin(['Consumer loans'], level='NAME_CONTRACT_TYPE')][
            ['DAYS_DECISION', 'AMT_CREDIT']]
        consumer_loans.index = consumer_loans.index.droplevel('NAME_CONTRACT_TYPE')
        cash_loans = df[df.index.isin(['Cash loans'], level='NAME_CONTRACT_TYPE')][['DAYS_DECISION', 'AMT_CREDIT']]
        cash_loans.index = cash_loans.index.droplevel('NAME_CONTRACT_TYPE')
        revolving_loans = df[df.index.isin(['Revolving loans'], level='NAME_CONTRACT_TYPE')][
            ['DAYS_DECISION', 'AMT_CREDIT']]
        revolving_loans.index = revolving_loans.index.droplevel('NAME_CONTRACT_TYPE')
        dfs = [consumer_loans, cash_loans, revolving_loans]
        df = reduce(lambda left, right: pd.merge(left, right, on='SK_ID_CURR', how='outer'), dfs)
        df.columns = ['Days_consumerloans', 'AMT_credit_consumerloans', 'Days_cashloans', 'AMT_credit_cashloans',
                      'Days_revolvingloans', 'AMT_credit_revolvingloans']
        prev1 = prev1.merge(df, on='SK_ID_CURR', how='outer')
        del prev, df
        gc.collect()
        return prev1

    def hc_credit_card(self):
        """
        This method reads credit_card_balance table that includes the recorded
        previous credit card transactions at Home Credit financial institution, perform manually feature engineering,
        flatten many transactions, and returns the statistics related to each application SK_ID_CURR.
        """

        print('Processing credit_card_balance table related to Home Credit source...')
        ccb = pd.read_csv('data/credit_card_balance.csv')
        ccb['Beyond_limit'] = np.where(ccb['AMT_BALANCE'] > ccb['AMT_CREDIT_LIMIT_ACTUAL'], 1, 0)
        ccb['Delinquent'] = np.where(ccb['AMT_PAYMENT_CURRENT'] < ccb['AMT_INST_MIN_REGULARITY'], 1, 0)

        # Percentage change of credit card balance between two successive months
        def _pct_diff(group):
            group['balance_pct_change'] = group['AMT_BALANCE'].diff() / (group['AMT_BALANCE'].shift(1) + 1)
            group['balance_pct_change'] = np.where(group['balance_pct_change'] > 30, 30, group['balance_pct_change'])
            return group

        ccb = ccb.sort_values('MONTHS_BALANCE').groupby('SK_ID_PREV').apply(_pct_diff)
        # how many times credit card reached 80% and above?
        ccb['times_bal80'] = np.where(ccb['AMT_BALANCE'] / ccb['AMT_CREDIT_LIMIT_ACTUAL'] >= 0.8, 1, 0)
        # How many credit card do applicant have at the moment?
        # What is the total balance on them? (outstanding debt of credit card)
        ccb['has_cc_now'] = np.where(ccb['MONTHS_BALANCE'] == -1, 1, 0)
        ccb['balance_now'] = ccb['has_cc_now'] * ccb['AMT_BALANCE']
        stats = ccb.groupby('SK_ID_CURR', as_index=False).agg(
            {'has_cc_now': 'sum', 'balance_now': 'sum', 'times_bal80': 'sum'})
        # Flatten last four months balance and percentage change for each applicant
        cols = ['MONTHS_BALANCE', 'Beyond_limit', 'Delinquent', 'balance_pct_change']
        cclast = ccb.sort_values('MONTHS_BALANCE', ascending=False).groupby(['SK_ID_CURR'])[cols].nth(0)
        cclag1 = ccb.sort_values('MONTHS_BALANCE', ascending=False).groupby(['SK_ID_CURR'])[cols].nth(1)
        cclag2 = ccb.sort_values('MONTHS_BALANCE', ascending=False).groupby(['SK_ID_CURR'])[cols].nth(2)
        cclag3 = ccb.sort_values('MONTHS_BALANCE', ascending=False).groupby(['SK_ID_CURR'])[cols].nth(3)
        dfs = [cclast, cclag1, cclag2, cclag3]
        lags = reduce(lambda left, right: pd.concat([left, right], axis=1, sort=False), dfs)
        columns = []
        for i in range(4):
            columns += ['MONTHS_BALANCE' + str(i), 'Beyond_limit' + str(i), 'Delinquent' + str(i),
                        'balance_pct_change' + str(i)]
        lags.columns = columns
        stats = stats.merge(lags, on='SK_ID_CURR', how='outer')
        del lags, ccb
        gc.collect()
        return stats

    def hc_installment(self):
        """
        This method reads installments_payments table that includes the recorded
        previous installments at Home Credit financial institution, perform manually feature engineering,
        flatten many transactions, and returns the statistics related to each application SK_ID_CURR.
        """

        print('Processing installments table related to Home Credit source...')
        insta = pd.read_csv('data/installments_payments.csv')
        # 'DAYS_INSTALMENT': days before credit card supposed to be paid,
        # 'DAYS_ENTRY_PAYMENT': days that amount was acutually paid.
        insta['insta_delinquency'] = np.where(insta['DAYS_INSTALMENT'] >= insta['DAYS_ENTRY_PAYMENT'], 0, 1)
        insta['insta_debt'] = insta['AMT_INSTALMENT'] - insta['AMT_PAYMENT']
        stats = insta.sort_values('DAYS_INSTALMENT', ascending=False).groupby(['SK_ID_CURR']).agg(
            {'DAYS_INSTALMENT': 'first', 'insta_debt': ['sum', 'mean', 'first'], 'insta_delinquency': ['sum', 'first']})
        stats.columns = stats.columns.map('_'.join)
        del insta
        gc.collect()
        return stats

    def hc_pos_cash(self):
        """
        This method reads POS_CASH_balance table that includes the recorded
        previous point of sale (POS) at Home Credit financial institution, perform manually feature engineering,
        flatten many transactions, and returns the statistics related to each application SK_ID_CURR.
        """

        print('Processing POS_CASH_balance table related to Home Credit source...')
        pc = pd.read_csv('data/POS_CASH_balance.csv')
        # Flatten all the columns for the latest 4 POS data for each application
        pc = pc.sort_values('MONTHS_BALANCE', ascending=False)
        cols = ['MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF']
        pos0 = pc.groupby(['SK_ID_CURR'])[cols].first()
        pos1 = pc.groupby(['SK_ID_CURR'])[cols].nth(1)
        pos2 = pc.groupby(['SK_ID_CURR'])[cols].nth(2)
        pos3 = pc.groupby(['SK_ID_CURR'])[cols].nth(3)
        data_frames = [pos0, pos1, pos2, pos3]
        poslag = reduce(lambda left, right: pd.concat([left, right], axis=1, sort=False), data_frames)
        columns = []
        for i in range(4):
            columns += ['MONTHS_BALANCE' + str(i), 'SK_DPD' + str(i), 'SK_DPD_DEF' + str(i)]
        poslag.columns = columns
        del pc, pos0, pos1, pos2, pos3
        gc.collect()
        return poslag

    def application_train(self):
        """
        This method reads application_train table that includes all the current applications, cleans it, and
        performs manually feature engineering
        """

        print('Processing application_train table for the current loan application')
        train = pd.read_csv('data/application_train.csv')
        # Delete four applications with XNA CODE_GENDER (train set)
        train = train[train['CODE_GENDER'] != 'XNA']
        # Replace DAYS_EMPLOYED = 365243 by nan
        train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
        # Feature engineering
        train['Days_employed_age'] = train['DAYS_EMPLOYED'] / train['DAYS_BIRTH']
        train['Credit_income_ratio'] = train['AMT_CREDIT'] / train['AMT_INCOME_TOTAL']
        train['Anuity_income_ratio'] = train['AMT_ANNUITY'] / train['AMT_INCOME_TOTAL']
        train['Income_per_person'] = train['AMT_INCOME_TOTAL'] / train['CNT_FAM_MEMBERS']
        # length of the payment in months since the annuity is the monthly amount due
        train['Credit_term'] = train['AMT_ANNUITY'] / train['AMT_CREDIT']
        return train

    def bureau(self):

        """
        This method reads bureau and bureau_balance tables that includes the recorded
        at Bureau, perform manually feature engineering,flatten transactions, and
        returns the statistics related to each application SK_ID_CURR.
        """

        print('Processing bureau and bureau balance tables...')
        bureau = pd.read_csv('data/bureau.csv')

        # ------------------------------------Feature Engineering (1): General ---------------------------------------
        bureau['Days_early_paidoff'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
        bureau['Duration_real'] = bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT']
        bureau['Duration_planned'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_CREDIT']
        # Replace data with Duration_planned = 0 with 1 to avoide devision by zero
        bureau['Duration_planned'].replace({0: 1}, inplace=True)
        # Weighted sum of credit a person borrowed (per days)
        bureau['AMT_weightdebt_duration'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['Duration_planned']
        # 'AMT_CREDIT_SUM_OVERDUE': Current amount overdue on credit
        bureau['AMT_Overdue_duration'] = bureau['AMT_CREDIT_SUM_OVERDUE'] / bureau['Duration_planned']
        # Maximal amount overdue so far
        bureau['AMT_Maxoverdue_duration'] = bureau['AMT_CREDIT_MAX_OVERDUE'] / bureau['Duration_planned']
        # Defaulted: CREDIT_DAY_OVERDUE > 270 days is considered defaluted
        bureau['Defaulted'] = np.where(bureau['CREDIT_DAY_OVERDUE'] > 270, 1, 0)
        bureau['AMT_defaulted'] = bureau['Defaulted'] * bureau['AMT_CREDIT_SUM_DEBT']
        # Encoding CREDIT_ACTIVE ('Closed','Active') to (0,1)
        mapping = {'Closed': 0, 'Active': 1}
        bureau['CREDIT_ACTIVE'] = bureau['CREDIT_ACTIVE'].map(mapping)
        # Flatten manual features with aggregations
        stats = bureau.sort_values('DAYS_CREDIT', ascending=False).groupby('SK_ID_CURR') \
            .agg({'AMT_CREDIT_SUM_DEBT': ['count', 'sum', 'mean'],
                  'AMT_weightdebt_duration': ['sum', 'mean'],
                  'AMT_Overdue_duration': ['sum', 'mean'],
                  'AMT_Maxoverdue_duration': ['mean'],
                  'Days_early_paidoff': ['sum', 'mean'],
                  'Defaulted': ['sum', 'mean'],
                  'AMT_defaulted': ['sum', 'mean'],
                  'CREDIT_ACTIVE': 'sum'})
        stats.columns = stats.columns.map('_'.join)
        # Flatten last four stats for each applicant (nth() method does not work with .agg method)
        columns = ['SK_ID_CURR', 'DAYS_CREDIT', 'AMT_CREDIT_SUM_DEBT', 'AMT_weightdebt_duration',
                   'AMT_Overdue_duration', \
                   'Days_early_paidoff', 'Defaulted', 'AMT_defaulted']
        stats0 = bureau.sort_values('DAYS_CREDIT', ascending=False)[columns].groupby('SK_ID_CURR').nth(0)
        stats1 = bureau.sort_values('DAYS_CREDIT', ascending=False)[columns].groupby('SK_ID_CURR').nth(1)
        stats2 = bureau.sort_values('DAYS_CREDIT', ascending=False)[columns].groupby('SK_ID_CURR').nth(2)
        stats3 = bureau.sort_values('DAYS_CREDIT', ascending=False)[columns].groupby('SK_ID_CURR').nth(3)
        data_frames = [stats0, stats1, stats2, stats3]
        lags = reduce(lambda left, right: pd.concat([left, right], axis=1, sort=False), data_frames)
        col = []
        for i in range(4):
            col += ['DAYS_CREDIT' + str(i), 'AMT_CREDIT_SUM_DEBT' + str(i), 'AMT_weightdebt_duration' + str(i), \
                    'AMT_Overdue_duration' + str(i), 'Days_early_paidoff' + str(i), 'Defaulted' + str(i),
                    'AMT_defaulted' + str(i)]
        lags.columns = col
        stats = pd.merge(stats, lags, on='SK_ID_CURR', how='outer')

        # ------------------------Feature Engineering (2): Loan cycle life for different CREDIT_TYPE--------------------
        # Select 6 categories of loans with the highest frequency
        filter = (bureau['CREDIT_TYPE'] == 'Mortgage') | (bureau['CREDIT_TYPE'] == 'Real estate loan') \
                 | (bureau['CREDIT_TYPE'] == 'Car loan') \
                 | (bureau['CREDIT_TYPE'] == 'Loan for business development') \
                 | (bureau['CREDIT_TYPE'] == 'Loan for the purchase of equipment') \
                 | (bureau['CREDIT_TYPE'] == 'Cash loan (non-earmarked)') \
                 | (bureau['CREDIT_TYPE'] == 'Loan for working capital replenishment')
        btype = bureau[filter].copy()
        # first(DAYS_CREDIT) is when the last credit of each credit type applied and
        # last(DAYS_CREDIT) is when the first onces applied.
        bt_stats = btype.sort_values('DAYS_CREDIT', ascending=False).groupby(['SK_ID_CURR', 'CREDIT_TYPE']) \
            .agg({'DAYS_CREDIT': ['first', 'last'], 'AMT_CREDIT_SUM_DEBT': ['count', 'sum', 'mean'], \
                  'CREDIT_ACTIVE': 'sum', 'Defaulted': 'sum', 'AMT_defaulted': ['sum']})
        bt_stats.columns = bt_stats.columns.map('_'.join)
        mortgage = bt_stats[bt_stats.index.isin(['Mortgage'], level='CREDIT_TYPE')]
        mortgage.index = mortgage.index.droplevel('CREDIT_TYPE')
        realestate = bt_stats[bt_stats.index.isin(['Real estate loan'], level='CREDIT_TYPE')]
        realestate.index = realestate.index.droplevel('CREDIT_TYPE')
        carloan = bt_stats[bt_stats.index.isin(['Car loan'], level='CREDIT_TYPE')]
        carloan.index = carloan.index.droplevel('CREDIT_TYPE')
        loanbusiness = bt_stats[bt_stats.index.isin(['Loan for business development'], level='CREDIT_TYPE')]
        loanbusiness.index = loanbusiness.index.droplevel('CREDIT_TYPE')
        loanpurchase = bt_stats[bt_stats.index.isin(['Loan for the purchase of equipment'], level='CREDIT_TYPE')]
        loanpurchase.index = loanpurchase.index.droplevel('CREDIT_TYPE')
        cashloan = bt_stats[bt_stats.index.isin(['Cash loan (non-earmarked)'], level='CREDIT_TYPE')]
        cashloan.index = cashloan.index.droplevel('CREDIT_TYPE')
        workingloan = bt_stats[bt_stats.index.isin(['Loan for working capital replenishment'], level='CREDIT_TYPE')]
        workingloan.index = workingloan.index.droplevel('CREDIT_TYPE')
        dataframes = [mortgage, realestate, carloan, loanbusiness, loanpurchase, cashloan, workingloan]
        credit_type = reduce(lambda left, right: pd.merge(left, right, on='SK_ID_CURR', how='outer'), dataframes)
        types = ['mortgage', 'realestate', 'carloan', 'loanbusiness', 'loanpurchase', 'cashloan', 'workingloan']
        columns = []
        for s in types:
            columns += ['DAYS_CREDIT_first_' + s, 'DAYS_CREDIT_last_' + s, 'AMT_CREDIT_SUM_DEBT_count_' + s, \
                        'AMT_CREDIT_SUM_DEBT_sum_' + s, 'AMT_CREDIT_SUM_DEBT_mean_' + s, \
                        'CREDIT_ACTIVE_sum_x_' + s, 'Defaulted_sum_x_' + s, 'AMT_defaulted_sum_' + s]
        credit_type.columns = columns
        stats.merge(credit_type, on='SK_ID_CURR', how='outer')

        # -----------------------------Feature Engineering (3): Loan cycle life for Credit Card-------------------------
        ccdebt_bureau = bureau[bureau['CREDIT_TYPE'] == 'Credit card'].sort_values('DAYS_CREDIT', ascending=False) \
            .groupby('SK_ID_CURR').agg({'AMT_CREDIT_SUM_DEBT': ['count', 'sum', 'mean', 'first']})
        ccdebt_bureau.columns = ccdebt_bureau.columns.map('_'.join)
        stats = stats.merge(ccdebt_bureau, on='SK_ID_CURR', how='outer')
        # ------------------------------Feature Engineering (4): Bureau Balance Table-----------------------------------
        bureaubal = pd.read_csv('data/bureau_balance.csv')
        # When did the credit closed? When was the last delinquency for each bureau credit?
        # Last close is obtained by first 0 in MONTHS_BALANCE, last delinquency is obtained from first 1.
        bbalance = bureaubal.groupby(['SK_ID_BUREAU', 'STATUS'], as_index=False).first()
        left = bbalance[(bbalance['STATUS'] == '0')][['SK_ID_BUREAU', 'MONTHS_BALANCE']]
        right = bbalance[(bbalance['STATUS'] == '1')][['SK_ID_BUREAU', 'MONTHS_BALANCE']]
        bbalance = pd.merge(left, right, on='SK_ID_BUREAU', how='left')
        bbalance.columns = ['SK_ID_BUREAU', 'Months_latest_open', 'Months_latest_delin']
        # Delinquency ratios: how often each bureau delayed?
        delinquency = pd.get_dummies(bureaubal[(bureaubal['STATUS'] != 'X') & (bureaubal['STATUS'] != 'C')])
        delinquency = delinquency.groupby('SK_ID_BUREAU').agg(
            {'STATUS_0': 'mean', 'STATUS_1': 'mean', 'STATUS_2': 'mean', 'STATUS_3': 'mean',
             'STATUS_4': 'mean', 'STATUS_5': 'mean'})
        bbalance = bbalance.merge(delinquency, on='SK_ID_BUREAU', how='inner')
        # Add SK_ID_CURR to bbalance dataframe
        bbalance = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].merge(bbalance, on='SK_ID_BUREAU', how='inner')
        # pick the latest open SK_ID_Bureau
        bbalance = bbalance.sort_values('Months_latest_open', ascending=False).groupby('SK_ID_CURR').first()
        # merge with stats
        stats = stats.merge(bbalance, on='SK_ID_CURR', how='outer')
        del bbalance, bureau, bureaubal, bt_stats, ccdebt_bureau
        gc.collect()
        return stats

    def merge_tables(self):
        '''
        This method merges the tables from bureau and home credit sources with the application_train table,
        and return one row for each SK_ID_CURR. Automated feature engineering and deep learning feature extraction
        are not included in this method.
        '''

        prev_home = self.hc_prv_app()
        prev_home = prev_home.merge(self.hc_credit_card(), on='SK_ID_CURR', how='outer')
        prev_home = prev_home.merge(self.hc_installment(), on='SK_ID_CURR', how='outer')
        prev_home = prev_home.merge(self.hc_pos_cash(), on='SK_ID_CURR', how='outer')
        train = self.application_train()
        train = train.merge(self.bureau(), on='SK_ID_CURR', how='left')
        train = train.merge(prev_home, on='SK_ID_CURR', how='left')
        # train.to_csv('merged_tables.csv')
        del prev_home
        gc.collect()
        return train

    def automated_features(self):
        '''
        This method performs automated feature engineering using feature tools package and returns a dataframe
        with added new features from all the tables. Important paramater of feature tools is max_depth of
        deep feature synthesis (typically set to 1 or 2).
        '''

        print('Performing automated feature engineering using feature tools')
        train = pd.read_csv('data/application_train.csv')
        bureau = pd.read_csv('data/bureau.csv')
        bureaubal = pd.read_csv('data/bureau_balance.csv')
        prev = pd.read_csv('data/previous_application.csv')
        ccb = pd.read_csv('data/credit_card_balance.csv')
        insta = pd.read_csv('data/installments_payments.csv')
        pc = pd.read_csv('data/POS_CASH_balance.csv')

        # Choosing nrows data from all datasets
        train = train.sample(frac=1)
        ids = train['SK_ID_CURR'].values
        bureau = bureau.loc[bureau['SK_ID_CURR'].isin(ids)]
        idsb = bureau['SK_ID_BUREAU'].values
        bureaubal = bureaubal.loc[bureaubal['SK_ID_BUREAU'].isin(idsb)]
        prev = prev.loc[prev['SK_ID_CURR'].isin(ids)]
        ccb = ccb.loc[ccb['SK_ID_CURR'].isin(ids)]
        insta = insta.loc[insta['SK_ID_CURR'].isin(ids)]
        pc = pc.loc[pc['SK_ID_CURR'].isin(ids)]

        # creating EntitySet (collection of tables)
        es = ft.EntitySet(id='applications')
        # adding Entity (table) to EntitySet
        es = es.entity_from_dataframe(entity_id='train', dataframe=train, index='SK_ID_CURR')
        es = es.entity_from_dataframe(entity_id='bureau', dataframe=bureau, index='SK_ID_BUREAU')
        es = es.entity_from_dataframe(entity_id='bureaubal', dataframe=bureaubal, make_index=True, index='bb_id')
        es = es.entity_from_dataframe(entity_id='prev', dataframe=prev, index='SK_ID_PREV')
        es = es.entity_from_dataframe(entity_id='ccb', dataframe=ccb, make_index=True, index='cc_id')
        es = es.entity_from_dataframe(entity_id='insta', dataframe=insta, make_index=True, index='installment.id')
        es = es.entity_from_dataframe(entity_id='pc', dataframe=pc, make_index=True, index='pos_cash_id')
        # Creating relation between Entities
        # Relationship between application training and bureau
        r_applications_bureau = ft.Relationship(es['train']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])
        es = es.add_relationship(r_applications_bureau)
        # Relationship between bureau and bureau balance
        r_bureau_bureaubal = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureaubal']['SK_ID_BUREAU'])
        es = es.add_relationship(r_bureau_bureaubal)
        # Relationship between application training and previous applications
        r_app_prev = ft.Relationship(es['train']['SK_ID_CURR'], es['prev']['SK_ID_CURR'])
        es = es.add_relationship(r_app_prev)
        # Relationship between previous applications with credit card balance, pos cash, and installments
        r_prev_cc = ft.Relationship(es['prev']['SK_ID_PREV'], es['ccb']['SK_ID_PREV'])
        es = es.add_relationship(r_prev_cc)
        r_prev_insta = ft.Relationship(es['prev']['SK_ID_PREV'], es['insta']['SK_ID_PREV'])
        es = es.add_relationship(r_prev_insta)
        r_prev_pc2 = ft.Relationship(es['prev']['SK_ID_PREV'], es['pc']['SK_ID_PREV'])
        es = es.add_relationship(r_prev_pc2)

        # Deep feature synthesis with depth of 2 by stacking feature primitives (aggregations and transformations)
        # Automated features are concatenated to the original features; Therefore,
        train_ft, feature_names = ft.dfs(entityset=es, target_entity='train', max_depth=self.ft_maxdep)
        train_ft = train_ft.reset_index()

        print('\nTotal number of features after adding automated features: ', train_ft.shape[1])
        del train, bureau, bureaubal, prev, ccb, insta, pc
        gc.collect()
        return train_ft

    def bureau_cnn_features(self):
        '''
        Convolution Neural Network (CNN) is used to extract new feature from sequential data from bureau and
        bureaubal tables. The method concatenats new features to applicaiton_train table and returns final dataframe.
        '''

        print('Extracting features using Convolutional Neural Network (CNN) ...')
        train = pd.read_csv('data/application_train.csv')
        idl = train['SK_ID_CURR'].values
        bureau = pd.read_csv('data/bureau.csv')
        # Imputating the missing data in bureau table
        # Missing categorical features are imputed with 'Not_applicable'
        # Missing numeric features are imputed with Zero (logical choice for this dataset)
        cols = bureau.select_dtypes(include=object).columns
        bureau[cols] = bureau[cols].fillna('Not_Applicable')
        cols = bureau.select_dtypes(exclude=object).columns
        bureau[cols] = bureau[cols].fillna(0)
        # One-hot encoding of categorical features
        bureau = pd.get_dummies(bureau, drop_first=True)
        bureau = bureau.sort_values('DAYS_CREDIT', ascending=False)
        lst = bureau['SK_ID_CURR'].values
        lst = list(set(lst))
        lst.sort()

        # Making bureau table data structure similar to an image
        # Applications are grouoped by SK_ID_CURR and for each SK_ID_CURR, the 5 most recent SK_ID_BUREAU is considered.
        # If an SK_ID_CURR did not have 5 records, empty rows added and filled with -99 (to avoid confusion with zero).
        group = bureau.groupby('SK_ID_CURR')
        b = []  # b is the reshaped data structure of bureau table, suitable for use in CNN
        j = 0
        for sk in idl:
            if sk in lst:
                a = group.get_group(lst[j])
                if a.shape[0] >= 5:
                    a = a[:5]
                else:
                    # m99 represents rows having value of -99
                    m99 = np.ones((5 - a.shape[0], a.shape[1])) * -99
                    m99 = pd.DataFrame(m99, columns=a.columns)
                    a = a.append(m99)
                a = a.drop(['SK_ID_CURR', 'SK_ID_BUREAU'], axis=1)
                a = a.values.flatten().tolist()
                b.extend(a)
                j += 1
            else:
                m99 = np.ones((5, bureau.shape[1])) * -99
                m99 = pd.DataFrame(m99, columns=bureau.columns)
                m99 = m99.drop(['SK_ID_CURR', 'SK_ID_BUREAU'], axis=1)
                m99 = m99.values.flatten().tolist()
                b.extend(m99)
        b = np.array(b)
        b = np.reshape(b, (idl.shape[0], 5, bureau.shape[1] - 2, 1))
        print('shape of channel(bureau):', b.shape)
        y = train['TARGET']
        y = to_categorical(y, 2)

        # Deep CNN implementation
        # CNN architecture includes 2 convolution layer followed by two fully connected layer
        np.random.seed(5)
        model = Sequential()

        # 1st conv layer
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding="same",
                         input_shape=(b.shape[1], b.shape[2], 1), data_format="channels_last"
                         ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # 2nd conv layer
        model.add(Conv2D(32, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # FC1
        model.add(Dense(units=128))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        # FC2
        model.add(Dense(units=100, name='feature_extract'))
        model.add(Activation("relu"))
        # output FC
        model.add(Dense(units=2, activation='sigmoid'))
        model.build()
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['AUC'])
        model.summary()

        # Train deep neural network
        early_stops = EarlyStopping(patience=5, monitor='val_auc')
        mc = ModelCheckpoint('best_model.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True)
        model.fit(b, y, validation_split=0.05,
                  callbacks=[early_stops, mc], batch_size=self.cnn_bsize, epochs=self.cnn_epoch, verbose=1)

        # Extract the useful featuer from CNN after training the deep nerual network
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer('feature_extract').output)
        intermediate_layer_model.summary()

        # predict to get featured data
        feauture_engg_data = intermediate_layer_model.predict(b)
        feauture_engg_data = pd.DataFrame(feauture_engg_data)
        print('feauture_engg_data shape:', feauture_engg_data.shape)

        # Renaming columns
        new_col = []
        for i in range(100):
            new_col.append('bfeat_%d' % (i + 1))
        feauture_engg_data.columns = new_col
        feauture_engg_data['SK_ID_CURR'] = idl
        train = train.merge(feauture_engg_data, on='SK_ID_CURR', how='left')
        del feauture_engg_data, bureau
        gc.collect()
        return train

    def bureau_rnn_features(self):
        '''
        Recurrent Neural Network (RNN) is used to extract new feature from sequential data from bureau and
        bureaubal tables. The method concatenats new features to applicaiton_train table and returns final dataframe.
        '''

        print('Extracting features using Recurrent Neural Network (RNN) ...')
        train = pd.read_csv('data/application_train.csv')
        idl = train['SK_ID_CURR'].values
        bureau = pd.read_csv('data/bureau.csv')
        # Imputating the missing data in bureau table
        # Missing categorical features are imputed with 'Not_applicable'
        # Missing numeric features are imputed with Zero (logical choice for this dataset)
        cols = bureau.select_dtypes(include=object).columns
        bureau[cols] = bureau[cols].fillna('Not_Applicable')
        cols = bureau.select_dtypes(exclude=object).columns
        bureau[cols] = bureau[cols].fillna(0)

        # One-hot encoding of categorical features
        bureau = pd.get_dummies(bureau, drop_first=True)
        bureau = bureau.sort_values('DAYS_CREDIT', ascending=False)
        lst = bureau['SK_ID_CURR'].values
        lst = list(set(lst))
        lst.sort()

        # Making bureau table data structure similar to an image
        # Applications are grouoped by SK_ID_CURR and for each SK_ID_CURR, the 5 most recent SK_ID_BUREAU is considered.
        # If an SK_ID_CURR did not have 5 records, empty rows added and filled with -99 (to avoid confusion with zero).
        group = bureau.groupby('SK_ID_CURR')
        b = []  # b is the reshaped data structure of bureau table, suitable for use in RNN
        j = 0
        for sk in idl:
            if sk in lst:
                a = group.get_group(lst[j])
                if a.shape[0] >= 5:
                    a = a[:5]
                else:
                    # m99 represents rows having value of -99
                    m99 = np.ones((5 - a.shape[0], a.shape[1])) * -99
                    m99 = pd.DataFrame(m99, columns=a.columns)
                    a = a.append(m99)
                a = a.drop(['SK_ID_CURR', 'SK_ID_BUREAU'], axis=1)
                a = a.values.flatten().tolist()
                b.extend(a)
                j += 1
            else:
                m99 = np.ones((5, bureau.shape[1])) * -99
                m99 = pd.DataFrame(m99, columns=bureau.columns)
                m99 = m99.drop(['SK_ID_CURR', 'SK_ID_BUREAU'], axis=1)
                m99 = m99.values.flatten().tolist()
                b.extend(m99)
        b = np.array(b)
        b = np.reshape(b, (idl.shape[0], 5, bureau.shape[1] - 2))
        print('shape of channel(bureau):', b.shape)
        y = train['TARGET']
        y = to_categorical(y, 2)

        # Deep RNN implementation
        # RNN architecture includes 2 Long Short Term Memory (LSTM) units followed by two fully connected layer
        np.random.seed(5)
        model = Sequential()
        # 1st LSTM layer
        model.add(LSTM(units=50, input_shape=(b.shape[1], b.shape[2]), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        # 2nd LSTM layer
        model.add(LSTM(50, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        model.add(Flatten())
        # FC1
        model.add(Dense(units=128))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        # FC2
        model.add(Dense(units=100, name='RNN_feature_extract'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        # output FC
        model.add(Dense(units=2, activation='sigmoid'))
        model.build()
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['AUC'])
        model.summary()

        # Train recurrent neural network
        early_stops = EarlyStopping(patience=5, monitor='val_auc')
        mc = ModelCheckpoint('best_model.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True)
        model.fit(b, y, validation_split=0.05, callbacks=[early_stops, mc], batch_size=self.rnn_bsize,
                  epochs=self.rnn_epoch,
                  verbose=1)

        # Extract the useful featuer from RNN after training the deep nerual network
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer('RNN_feature_extract').output)
        intermediate_layer_model.summary()
        # predict to get featured data
        feauture_engg_data = intermediate_layer_model.predict(b)
        feauture_engg_data = pd.DataFrame(feauture_engg_data)
        print('feauture_engg_data shape:', feauture_engg_data.shape)

        # Renaming columns
        new_col = []
        for i in range(100):
            new_col.append('bfeat_%d' % (i + 1))
        feauture_engg_data.columns = new_col
        feauture_engg_data['SK_ID_CURR'] = idl

        # Merge RNN features to application_train dataset
        train = train.merge(feauture_engg_data, on='SK_ID_CURR', how='left')
        del feauture_engg_data, bureau
        gc.collect()
        return train

    def XGBoost(self):
        '''
        This method train a machine learning model using XGBoost algorithm. Before that, it imputes the empty cells
        in self.data table, encodes categorical features using one-hot encoding method, applies PCA transformation
        on first self.pca_n principles components.

        Returns:
        self.pred_class: Binary class prediction of the target variable.
        self.pred: Probability prediction of the target variable.
        self.y_test: y_test in the training dataset
        '''

        print('Preprocessing final table one-hot encoding categorical features...')
        # Drop the columns with correlation > 0.98
        corr = self.data.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
        self.data = self.data.drop(to_drop, axis=1)

        # Imputating the missing data, PCA can not handle missing data
        # Missing categorical features are imputed with 'Not_applicable'
        # Missing numeric features are imputed with Zero (logical choice for this dataset)
        cols = self.data.select_dtypes(include=object).columns
        self.data[cols] = self.data[cols].fillna('Not_Applicable')
        cols = self.data.select_dtypes(exclude=object).columns
        self.data[cols] = self.data[cols].fillna(0)

        # One-hot encoding categorical features for XGBoost algorithm.
        self.data = pd.get_dummies(self.data, drop_first=True)

        # Train and test set split
        y = self.data['TARGET']
        X = self.data.drop('TARGET', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=1234)
        self.y_test = y_test

        # First pca_n principle components are used for XGBoost
        # PCA requires standardization of features
        sc = StandardScaler()
        X_pca = sc.fit_transform(X_train)
        pca = PCA(n_components=self.pca_n)
        X_pca = pca.fit_transform(X_pca)
        print('\nRatio of variance explained by {} principal components: '.format(self.pca_n),
              sum(pca.explained_variance_ratio_))

        pipeline = make_pipeline(StandardScaler(), PCA(n_components=self.pca_n), XGBClassifier())
        params = {
            'xgbclassifier__learning_rate': [0.05, 0.1, 0.15, 0.2],
            'xgbclassifier__max_depth': [3, 4, 5, 6, 8, 10],
            'xgbclassifier__min_child_weight': [1, 3, 5, 7],
            'xgbclassifier__gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'xgbclassifier__colsample_bytree': [0.5, 0.7, 1]
        }
        print('\nApplying XGBoost classifier... \n')
        model = RandomizedSearchCV(pipeline, params, n_iter=1, scoring='roc_auc', cv=self.nfolds, n_jobs=-1, verbose=3)
        model.fit(X_train, y_train)
        print('\nCross validation best score(AUC) is:', model.best_score_)
        # Hyperparameters of the model with the best performance
        print('\nModel best hyperparamters are:', model.best_params_)
        # Binary class prediction
        self.pred_class = model.predict(X_test)
        # Probability prediction
        self.pred = model.predict_proba(X_test)
        self.pred = [p[1] for p in self.pred]

    def lightGBM(self):
        '''
        This method trains a machine learning model using LightGBM algorithm. The boosted algorithm hyper parameters
        was found using Bayasian optimization. This methods encodes categorical features as integer
        and save them as 'category' type for lightGBM algorithms.

        Returns:
        self.pred_class: Binary class prediction of the target variable.
        self.pred: Probability prediction of the target variable.
        self.y_test: y_test in the training dataset
        '''

        print('Preprocessing final table and label encoding categorical features...')
        # Drop the columns with correlation > 0.98
        corr = self.data.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
        self.data = self.data.drop(to_drop, axis=1)
        # Encoding categorical features because lightGBM offers good accuracy with integer-encoded categorical features.
        class_le = LabelEncoder()
        cols = self.data.select_dtypes(include=object).columns
        for col in cols:
            self.data[col] = class_le.fit_transform(self.data[col].values.astype(str))
            self.data[col] = self.data[col].astype('category')
        print('Applying LightGBM algorithm...')
        y = self.data['TARGET']
        X = self.data.drop('TARGET', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=1234)
        self.y_test = y_test
        # Lighgbm parameters was found using Bayesian Optimization
        model_params = {
            'colsample_bytree': 0.45544541538547634,
            'learning_rate': 0.09712737568777673,
            'max_depth': 10,
            'min_child_weight': 44.81416318834993,
            'min_split_gain': 0.47913323843650946,
            'num_leaves': 44,
            'reg_alpha': 8.507126649843658,
            'reg_lambda': 2.2113739093853257,
            'subsample': 0.43342993037373423
        }
        model = make_pipeline(StandardScaler(), LGBMClassifier(**model_params))
        # cross validation scores
        scores = cross_val_score(model, X_train, y_train, scoring='roc_auc', cv=self.nfolds, n_jobs=-1, verbose=100)
        print('max cross_val AUC: ', np.max(scores))
        model.fit(X_train, y_train)
        # Binary class prediction
        self.pred_class = model.predict(X_test)
        # Probability prediction
        self.pred = model.predict_proba(X_test)
        self.pred = [p[1] for p in self.pred]

    def Catboost(self):
        '''
        This methods trains a machine learning model using Catboost algorithm.
        This method encodes categorical features and save them as 'category' type for Catboost
        algorithm.

        Returns:
        self.pred_class: Binary class prediction of the target variable.
        self.pred: Probability prediction of the target variable.
        self.y_test: y_test in the training dataset.
        '''

        print('Preprocessing final table and one-hot encoding categorical features...')
        # Drop the columns with correlation > 0.98
        corr = self.data.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
        self.data = self.data.drop(to_drop, axis=1)
        # Encoding categorical features because Catboost offers good accuracy with integer-encoded categorical features.
        class_le = LabelEncoder()
        cols = self.data.select_dtypes(include=object).columns
        for col in cols:
            self.data[col] = class_le.fit_transform(self.data[col].values.astype(str))
            self.data[col] = self.data[col].astype('category')

        print('Applying CatBoost algorithm...')
        y = self.data['TARGET']
        X = self.data.drop('TARGET', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=1234)
        self.y_test = y_test
        pipeline = Pipeline(steps=[('sc', StandardScaler()), ('catboost', CatBoostClassifier())])
        params = {
            "catboost__depth": [5, 6],
            "catboost__iterations": [500, 1000],
            "catboost__learning_rate": [0.001, 0.01, 0.1],
            "catboost__l2_leaf_reg": [5, 100]
        }
        model = RandomizedSearchCV(pipeline, params, n_iter=4, scoring='roc_auc', cv=self.nfolds, n_jobs=-1, verbose=3)
        model.fit(X_train, y_train)

        # Binary class prediction
        self.pred_class = model.predict(X_test)
        # Probability prediction
        self.pred = model.predict_proba(X_test)
        self.pred = [p[1] for p in self.pred]

    def FCNN(self):
        '''
        This method employs a fully connected neural network as the binary classifier.
        It impute the mission data and applies one-hot encoding on categorical features.

        Returns:
        self.pred_class: Binary class prediction of the target variable.
        self.pred: Probability prediction of the target variable.
        self.y_test: y_test in the training dataset.
        '''

        print('Preprocessing final table, imputing missing values, and One-hot encoding of categorical features...')
        # Drop the columns with correlation > 0.98
        corr = self.data.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
        self.data = self.data.drop(to_drop, axis=1)

        # Missing categorical features are imputed with 'Not_applicable'
        # Missing numeric features are imputed with Zero (logical choice for this dataset)
        cols = self.data.select_dtypes(include=object).columns
        self.data[cols] = self.data[cols].fillna('Not_Applicable')
        cols = self.data.select_dtypes(exclude=object).columns
        self.data[cols] = self.data[cols].fillna(0)

        # One-hot encoding of categorical features
        self.data = pd.get_dummies(self.data, drop_first=True)

        print('Applying Fully Connected Neural Network (FCNN) ...')
        y = self.data['TARGET']
        X = self.data.drop('TARGET', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=1234)
        self.y_test = y_test
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.fit_transform(X_test)

        # Deep FCNN implementation
        # FCNN architecture includes 3 fully connected units having 150, 75, 25 neurons, respectively.
        np.random.seed(5)
        # FC1
        model = Sequential()
        model.add(Dense(input_shape=(X_train.shape[1],), units=150))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        # FC2
        model.add(Dense(units=75))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        # FC3
        model.add(Dense(units=25))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        # Output layer
        model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
        model.build()
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['AUC'])
        model.summary()

        # Train deep neural network
        early_stops = EarlyStopping(patience=10, monitor='val_auc')
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=0, save_best_only=True)
        model.fit(X_train, y_train, validation_split=self.test_size, callbacks=[early_stops, mc],
                  batch_size=self.batch_size, epochs=self.epoch, verbose=1)
        # Binary class prediction, Keras predict method always return probability (unlike Sklearn!)
        self.pred_class = np.argmax(model.predict(X_test), axis=-1)
        # Probability prediction
        self.pred = model.predict(X_test)

    def Hclustering(self):
        '''
        This method resamples trining data to have balanced positive to negative labels. It undersamples the
        majority class (negative) using hierarcical clustering with cluster size equals to the size of positive class.
        Then randomly chose a sample from each cluster as the representative of that cluster.

        Returns:
        self.pred_class: Binary class prediction of the target variable.
        self.pred: Probability prediction of the target variable.
        self.y_test: y_test in the training dataset.
        '''

        print('Preprocessing final table and one-hot encoding categorical features... \n')
        # Drop the columns with correlation > 0.98
        corr = self.data.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
        self.data = self.data.drop(to_drop, axis=1)
        # Impute the missing data which is required for calculating euclidean distance in clustering
        # Missing value in categorical columns are imputed by NA (not available),
        # Missing value in numeric columns are imputed by zero, which most of the time is the case in this dataset.
        cols = self.data.select_dtypes(include=object).columns
        self.data[cols] = self.data[cols].fillna('NA')
        cols = self.data.select_dtypes(exclude=object).columns
        self.data[cols] = self.data[cols].fillna(0)

        # One-hot encoding categorical features for clustering (eucliden distance) and XGBoost algorithm.
        self.data = pd.get_dummies(self.data, drop_first=True)
        del corr
        y = self.data['TARGET']
        X = self.data.drop('TARGET', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=1234)
        self.y_test = y_test

        print('Undersampling the majority class using hierarchical clustering .... \n')
        # defining positive and negative classes
        negative = self.data.loc[self.data['TARGET'] == 0]
        positive = self.data.loc[self.data['TARGET'] == 1]
        # finding number of clusters
        nclusters = np.ceil(len(positive)).astype(int)
        # Standardize negative class for the clustering
        sc = StandardScaler()
        transform = sc.fit_transform(negative)
        negative = pd.DataFrame(transform, columns=negative.columns)
        # Clustering the majority class using euclidean affinity and ward linkage
        # In Ward's linkage, two clusters that lead to the minimum increase of the total within-cluster SSE are merged.
        ac = AgglomerativeClustering(n_clusters=nclusters, affinity='euclidean', linkage='ward')
        clustering = ac.fit(negative)
        # add a new feature for each row to show what cluster they belong
        negative['cluster'] = clustering.labels_
        # Randomly choose a sample from each cluster
        # A function for choosing one sample from each cluster
        def sampling(group):
            return group.sample(1, random_state=1)

        # Grouping the train data based on the cluster and select one sample from each cluster
        negative = negative.groupby('cluster', as_index=False).apply(sampling)
        negative = negative.droplevel(level=1)
        negative = negative.drop('cluster', axis=1)
        negative = pd.DataFrame(sc.inverse_transform(negative), columns=negative.columns)

        # Merging negative and positive class to form balanced train set
        train = pd.concat([negative, positive], axis=0, sort=False)
        # train['SK_ID_CURR'] = train['SK_ID_CURR'].astype(int)
        train = train.sample(frac=1)
        train = train.reset_index(drop=True)
        # Return X, y training dataset
        y_train = train['TARGET']
        X_train = train.drop('TARGET', axis=1)
        X_train = X_train.to_numpy()

        print('Applying XGBoost algorithm on balanced dataset... \n')
        pipeline = make_pipeline(StandardScaler(), PCA(n_components=self.pca_n), XGBClassifier())
        params = {
            'xgbclassifier__learning_rate': [0.05, 0.1, 0.15, 0.2],
            'xgbclassifier__max_depth': [3, 4, 5, 6, 8, 10],
            'xgbclassifier__min_child_weight': [1, 3, 5, 7],
            'xgbclassifier__gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'xgbclassifier__colsample_bytree': [0.5, 0.7, 1]
        }
        model = RandomizedSearchCV(pipeline, params, n_iter=1, scoring='roc_auc', cv=self.nfolds, n_jobs=-1, verbose=3)
        model.fit(X_train, y_train)
        print('\nCross validation best score(AUC) is:', model.best_score_)
        # Hyperparameters of the model with the best performance
        print('\nModel best hyperparamters are:', model.best_params_)
        # Binary class prediction
        self.pred_class = model.predict(X_test)
        # Probability prediction
        self.pred = model.predict_proba(X_test)
        self.pred = [p[1] for p in self.pred]
        del negative, positive, train, clustering, transform
        gc.collect()

    def Hclustering_smote(self):
        '''
        This method resamples training data to have balanced positive to negative labels. It undersamples the
        majority class (negative) using hierarcical clustering up to 50% of total data and oversamples the minority
        class up to 50% using SMOTE.

        Returns:
        self.pred_class: Binary class prediction of the target variable.
        self.pred: Probability prediction of the target variable.
        self.y_test: y_test in the training dataset.
        '''

        print('Preprocessing final table and one-hot encoding categorical features... \n')
        # Drop the columns with correlation > 0.98
        corr = self.data.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
        self.data = self.data.drop(to_drop, axis=1)
        # Impute the missing data which is required for calculating euclidean distance in clustering
        # Missing value in categorical columns are imputed by NA (not available),
        # Missing value in numeric columns are imputed by zero, which most of the time is the case in this dataset.
        cols = self.data.select_dtypes(include=object).columns
        self.data[cols] = self.data[cols].fillna('NA')
        cols = self.data.select_dtypes(exclude=object).columns
        self.data[cols] = self.data[cols].fillna(0)

        # One-hot encoding categorical features for clustering (eucliden distance) and XGBoost algorithm.
        self.data = pd.get_dummies(self.data, drop_first=True)

        # Agglomerative Clustering is computationaly expensive,
        # In this experiment only a fraction of application train file is considered (nrows= 30000).
        del corr
        y = self.data['TARGET']
        X = self.data.drop('TARGET', axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=1234)
        self.y_test = y_test

        print('Undersample majority class and oversampling minority class ...\n')

        # defining positive and negative classes
        negative = self.data.loc[self.data['TARGET'] == 0]
        positive = self.data.loc[self.data['TARGET'] == 1]

        # finding number of clusters (half of the training data)
        nclusters = np.ceil(len(self.data) / 2).astype(int)
        # Standardize negative class for the clustering
        sc = StandardScaler()
        transform = sc.fit_transform(negative)
        negative = pd.DataFrame(transform, columns=negative.columns)
        del transform

        # Clustering the majority class using euclidean affinity and ward linkage
        # In Ward's linkage, two clusters that lead to the minimum increase of the total within-cluster SSE are merged.
        ac = AgglomerativeClustering(n_clusters=nclusters, affinity='euclidean', linkage='ward')
        clustering = ac.fit(negative)
        # add a new feature for each row to show what cluster they belong
        negative['cluster'] = clustering.labels_
        # Randomly choose a sample from each cluster
        # A function for choosing one sample from each cluster
        def sampling(group):
            return group.sample(1, random_state=1)
        # Grouping the train data based on the cluster and select one sample from each cluster
        negative = negative.groupby('cluster', as_index=False).apply(sampling)
        negative = negative.droplevel(level=1)
        negative = negative.drop('cluster', axis=1)
        negative = pd.DataFrame(sc.inverse_transform(negative), columns=negative.columns)

        # Merging negative and positive class to form balanced train set
        train = pd.concat([negative, positive], axis=0, sort=False)
        train = train.sample(frac=1)
        train = train.reset_index(drop=True)

        # Return X, y training dataset
        y_train = train['TARGET']
        X_train = train.drop('TARGET', axis=1)

        # SMOTE oversampling of minority class
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        print('Applying XGBoost algorithm on the balanced training data... \n')
        pipeline = make_pipeline(StandardScaler(), PCA(n_components=self.pca_n), XGBClassifier())
        params = {
            'xgbclassifier__learning_rate': [0.05, 0.1, 0.15, 0.2],
            'xgbclassifier__max_depth': [3, 4, 5, 6, 8, 10],
            'xgbclassifier__min_child_weight': [1, 3, 5, 7],
            'xgbclassifier__gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'xgbclassifier__colsample_bytree': [0.5, 0.7, 1]
        }
        model = RandomizedSearchCV(pipeline, params, n_iter=1, scoring='roc_auc', cv=self.nfolds, n_jobs=-1, verbose=3)
        model.fit(X_train, y_train)
        print('\nCross validation best score(AUC) is:', model.best_score_)
        # Hyperparameters of the model with the best performance
        print('\nModel best hyperparamters are:', model.best_params_)
        # Binary class prediction
        self.pred_class = model.predict(X_test)
        # Probability prediction
        self.pred = model.predict_proba(X_test)
        self.pred = [p[1] for p in self.pred]

    def train(self, args):
        '''
            This method train the model using the selected algorithm (default: lightGBM).
            If resampling method is used, separate methods are implemented due to different pipeline structure.
        '''

        if args.resample:
            if args.use_hclstr:
                self.Hclustering()
            else:
                self.Hclustering_smote()
        elif args.xgb:
            print('using XGBoost for training ...')
            self.XGBoost()
        elif args.catb:
            print('using Catboost for training ...')
            self.Catboost()
        elif args.fcnn:
            print('Using Fully connected neural network for training ...')
            self.FCNN()
        else:
            print('using LightGBM for training ...')
            self.lightGBM()

        # Evaluate ROC_AUC, Precision, Recall, F1-Score, Cohen-Cappa metrics
        self.calculate_metrics()
        # Plot ROC curve
        self.plot_ROC()
        # Plot Precision/R curve
        self.plot_precision_recall()

    def calculate_metrics(self):
        '''
        This method calculates the classification metrics including precision, recall, F1-Score, AUC_ROC,
        and Cohen's kappa coefficient.
        '''

        # ROC_AUC score
        print('ROC_AUC:', roc_auc_score(self.y_test, self.pred))
        # Precision/Recall (0.1 Threshold)
        pred_class_2 = (np.array(self.pred) > 0.1).astype(int)
        cm = confusion_matrix(self.y_test, pred_class_2)
        print('\nConfusion_metrix (0.1 Threshold): \n', cm)
        # True Negatives (TN)
        tn = cm[0][0]
        # False Positives (FP)
        fp = cm[0][1]
        # False Negatives (FN)
        fn = cm[1][0]
        # True Positives (TP)
        tp = cm[1][1]
        precision = tp / (tp + fp)
        print('Precision (0.1 Threshold): ', precision)
        recall = tp / (tp + fn)
        print('Recall (0.1 Threshold): ', recall)
        print('F1-score ( 0.1 Threshold):', 2 * precision * recall / (precision + recall))
        cohen_kappa = cohen_kappa_score(self.y_test, pred_class_2)
        print('\nCohen_kappa (0.1 Threshold): ', cohen_kappa)

    def plot_ROC(self):
        '''
        This method plots ROC based on y_test and predicted probability of positive class by lightGBM.
        '''

        # Initialize figure
        fig = plt.figure(figsize=(9, 9))
        plt.title('Receiver Operating Characteristic')
        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.pred)
        plt.plot(fpr, tpr)
        # Diagonal 45 degree line
        plt.plot([0, 1], [0, 1], 'k--')
        # Axes limits and labels
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def plot_precision_recall(self):
        '''
        This method plots precision_recall curve based on y_test and predicted probability of positive class.
        '''

        precision, recall, thresholds = precision_recall_curve(self.y_test, self.pred)
        fig = plt.figure(figsize=(9, 9))
        plt.title('Precision_Recall')
        # Plot Precision-Recall curve
        plt.plot(recall, precision)
        # Axes limits and labels
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.show()
