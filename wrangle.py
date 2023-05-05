#######IMPORTS

import pandas as pd
import os
import env


#######FUNCTIONS


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
    