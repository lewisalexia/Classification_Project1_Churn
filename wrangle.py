# IMPORTS
import pandas as pd 
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from pydataset import data

import env
import os

from sklearn.model_selection import train_test_split


# ACQUIRE FUNCTIONS


def get_connection(db):
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'



def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe (from Misty Garcia, thank you!)
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df 




def get_telco_churn():
    """This function pings mySQL to grab the Telco data from the database using
    env.py credentials and url function.
    """
    url = env.get_connection('telco_churn')
    query = ''' select * from customers
	join contract_types
		using (contract_type_id)
	join internet_service_types
		using (internet_service_type_id)
	join payment_types
		using (payment_type_id)
        '''
    filename = 'telco_churn.csv'
    df = check_file_exists(filename, query, url)
    print(f'Load in successful, awaiting commands...')
    return df



# PREPARE FUNCTIONS



def clean_telco(df):
    '''This function will clean the telco_churn data'''
    
    #convert total charges to float
    df.total_charges = pd.to_numeric(df['total_charges'], errors='coerce')

    # fill n/a values in total_charges
    df.total_charges.fillna(df.monthly_charges)
    
    # dropping columns
    df = df.drop(columns=['contract_type_id','internet_service_type_id','payment_type_id'])

    # encoding
    df.gender = pd.get_dummies(df[['gender']], drop_first=True)

    df.partner = df.partner.replace('Yes',1).replace('No',0)

    df.dependents = df.dependents.replace('Yes',1).replace('No',0)

    df.phone_service = df.phone_service.replace('Yes',1).replace('No',0)

    df.online_security = df.online_security.replace('Yes',1).replace('No',0).replace('No internet service',0)

    df.online_backup = df.online_backup.replace('Yes',1).replace('No',0).replace('No internet service',0)

    df.device_protection = df.device_protection.replace('Yes',1).replace('No',0).replace('No internet service',0)

    df.tech_support = df.tech_support.replace('Yes',1).replace('No',0).replace('No internet service',0)

    df.streaming_tv = df.streaming_tv.replace('Yes',1).replace('No',0).replace('No internet service',0)

    df.streaming_movies = df.streaming_movies.replace('Yes',1).replace('No',0).replace('No internet service',0)

    df.paperless_billing = df.paperless_billing.replace('Yes',1).replace('No',0)

    df.churn = df.churn.replace('Yes',1).replace('No',0)

    df.multiple_lines = df.multiple_lines.replace('No phone service',0).replace('Yes',1).replace('No',0)

    # get dummies for categoricals
    dummy_df = pd.get_dummies(df[['contract_type','payment_type','internet_service_type']], dummy_na=False, drop_first=[True, True, True])

    # clean up and return final product
    df = pd.concat([df, dummy_df], axis=1)

    return df



# SPLIT FUNCTION



def split_telco(df):
    '''
    Takes in the telco dataframe and return train, validate, test subset dataframes
    and shapes to verify proper size split.
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify=df.churn)
    train, validate = train_test_split(train, #second split
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train.churn)
    
    print(train.shape, validate.shape, test.shape)

    return train, validate, test