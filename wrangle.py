import env
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

def get_data(directory=os.getcwd(), filename="zillow.csv"):
    """
    This function will:
    - Check local directory for csv file
        - return if exists
    - If csv doesn't exists:
        - create a df of the SQL_query
        - write df to csv
    - Output zillow df
    """
    SQL_query = "select * from properties_2017 pr left join predictions_2017 pred using(parcelid)"
    if os.path.exists(directory):
        df = pd.read_csv(filename) 
        return df
    else:
        url = env.get_db_url("zillow")

        df = pd.read_sql(SQL_query, url)

        #want to save to csv
        df.to_csv(filename)
        return df


def remove_columns(df, cols_to_remove):
    """
    Remove specified columns from a dataframe.
    
    Args:
        df (pandas.DataFrame): The input dataframe.
        cols_to_remove (list or str): A list of column names or a single column name to be removed from the dataframe.
    
    Returns:
        pandas.DataFrame: The dataframe with the specified columns removed.
    """
    df = df.drop(columns=cols_to_remove)
    return df


def analyze_missing_values(df):
    """
    Analyzes missing values in a dataframe and returns a summary dataframe.

    Args:
        df (pandas.DataFrame): The input dataframe containing observations and attributes.

    Returns:
        pandas.DataFrame: A dataframe with information about missing values for each attribute.
            The index represents attribute names, the first column contains the number of rows
            with missing values for that attribute, and the second column contains the percentage
            of total rows that have missing values for that attribute.
    """
    missing_counts = df.isnull().sum()
    total_rows = len(df)
    missing_percentages = (missing_counts / total_rows) * 100
    
    missing_data_df = pd.DataFrame({'Missing Count': missing_counts, 'Missing Percentage': missing_percentages})
    missing_data_df.index.name = 'Attribute'
    
    return missing_data_df



def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    """
    Drops rows and columns from a dataframe based on the proportion of missing values.

    Args:
        df (pandas.DataFrame): The input dataframe.
        prop_required_column (float, optional): The proportion of non-missing values required for each column.
            Defaults to 0.5.
        prop_required_row (float, optional): The proportion of non-missing values required for each row.
            Defaults to 0.75.

    Returns:
        pandas.DataFrame: The modified dataframe with dropped columns and rows.

    Raises:
        None

    Example:
        modified_df = handle_missing_values(df, prop_required_column=0.6, prop_required_row=0.8)
    """
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df