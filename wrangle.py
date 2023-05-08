#######IMPORTS

import pandas as pd
import os
import env


#######FUNCTIONS

zillow_query = """
        select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt,
        taxamount, fips
        from properties_2017
        where propertylandusetypeid = '261';
        """

def new_zillow_data(SQL_query):
    """
    This function will:
    - take in a SQL_query
    - create a db_url to mySQL
    - return a df of the given query from the telco_db
    """
    url = env.get_db_url('zillow')
    
    return pd.read_sql(SQL_query, url)

def get_zillow_data(SQL_query, filename = 'zillow.csv'):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - if csv doesn't exist:
        - creates df of sql query
        - writes df to csv: defaulted to telco.csv
    - outputs iris df
    """
    
    if os.path.exists(filename): 
        df = pd.read_csv(filename)
        return df
    else:
        df = new_zillow_data(SQL_query)

        df.to_csv(filename)
        return df

def wrangle_zillow(df):
    
    df.drop('Unnamed: 0', axis=1, inplace=True)
    
    df.rename(columns={'calculatedfinishedsquarefeet': 'squarefeet', 'taxvaluedollarcnt': 'taxvalue',
                       'fips': 'county'}, inplace=True)
    
    df.dropna(inplace=True)
    
    df[['bedroomcnt', 'squarefeet', 'taxvalue', 'yearbuilt', 'county']] = df[['bedroomcnt', 'squarefeet',
                                                                              'taxvalue', 'yearbuilt',
                                                                              'county']].astype(int)

    df.county = df.county.map({6037:'LA',6059:'Orange',6111:'Ventura'})
    
    df = df [df.squarefeet < 25_000]
    
    df = df [df.taxvalue < df.taxvalue.quantile(.95)].copy()
    
    return df



def split_data(df, stratify_col):
    '''
    Takes in two arguments the dataframe name and the ("name" - must be in string format) to stratify  and
    return train, validate, test subset dataframes will output train, validate, and test in that order
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2,
                                   random_state=123,
                                   stratify=df[stratify_col])
    train, validate = train_test_split(train, #second split
                                    test_size=.25,
                                    random_state=123,
                                    stratify=train[stratify_col])
    return train, validate, test







    