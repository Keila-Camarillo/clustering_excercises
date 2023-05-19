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
    SQL_query = '''
        SELECT prop.* ,
            pred.logerror,
            pred.transactiondate,
            air.airconditioningdesc as ac_type,
            arch.architecturalstyledesc archtectural_style,
            build.buildingclassdesc as bldg_class,
            heat.heatingorsystemdesc as heating,
            land.propertylandusedesc property_type,
            story.storydesc as story,
            type.typeconstructiondesc as construction_type
        from properties_2017 prop
        JOIN ( -- used to filter all properties with their last transaction date in 2017, w/o dups
                SELECT parcelid, MAX(transactiondate) AS max_transactiondate
                FROM predictions_2017
                GROUP BY parcelid) trans using (parcelid)
        -- bringing in logerror & transaction_date cols
        JOIN predictions_2017 pred ON trans.parcelid = pred.parcelid
                          AND trans.max_transactiondate = pred.transactiondate
        -- bringing in all other fields related to the properties
        LEFT JOIN airconditioningtype air USING(airconditioningtypeid)
        LEFT JOIN architecturalstyletype arch USING(architecturalstyletypeid)
        LEFT JOIN buildingclasstype build USING(buildingclasstypeid)
        LEFT JOIN heatingorsystemtype heat USING(heatingorsystemtypeid)
        LEFT JOIN propertylandusetype land USING(propertylandusetypeid)
        LEFT JOIN storytype story USING(storytypeid)
        LEFT JOIN typeconstructiontype type USING(typeconstructiontypeid)
        -- exercise stipulations
        WHERE propertylandusedesc = "Single Family Residential"
            AND transactiondate <= '2017-12-31'
            AND prop.longitude IS NOT NULL
            AND prop.latitude IS NOT NULL
        '''
    if os.path.exists(directory + filename):
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

def nulls_by_col(df):
    """
    This function will:
        - take in a dataframe
        - assign a variable to a Series of total row nulls for ea/column
        - assign a variable to find the percent of rows w/nulls
        - output a df of the two variables.
    """
    num_missing = df.isnull().sum()
    pct_miss = (num_missing / df.shape[0]) * 100
    cols_missing = pd.DataFrame({
                    'num_rows_missing': num_missing,
                    'percent_rows_missing': pct_miss
                    })
    
    return  cols_missing

def nulls_by_row(df, index_id = 'id'):
    """
    """
    num_missing = df.isnull().sum(axis=1)
    pct_miss = (num_missing / df.shape[1]) * 100
    
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': pct_miss})

    rows_missing = df.merge(rows_missing,
                        left_index=True,
                        right_index=True).reset_index()[[index_id, 'num_cols_missing', 'percent_cols_missing']]
    
    return rows_missing.sort_values(by='num_cols_missing', ascending=False)

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols



def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe) 
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # .value_counts()
    # observation of nulls in the dataframe
    # distribution of numerical attributes
    '''
    print(f"""SUMMARY REPORT
=====================================================
          
          
Dataframe head: 
{df.head(3)}
          
=====================================================
          
          
Dataframe info: """)
    df.info()

    print(f"""=====================================================
          
          
Dataframe Description: 
{df.describe().T}
          
=====================================================


nulls in dataframe by column: 
{nulls_by_col(df)}
=====================================================


nulls in dataframe by row: 
{nulls_by_row(df)}
=====================================================
    
    
DataFrame value counts: 
 """)         
    for col in (get_object_cols(df)): 
        print(f"""******** {col.upper()} - Value Counts:
{df[col].value_counts()}
    _______________________________________""")                   
        
    fig, axes = plt.subplots(1, len(get_numeric_cols(df)), figsize=(15, 5))
    
    for i, col in enumerate(get_numeric_cols(df)):
        sns.histplot(df[col], ax = axes[i])
        axes[i].set_title(f'Histogram of {col}')
    plt.show()

def outlier(df, feature, m=1.5):
    '''
    outlier will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    - using the `m` multiplier
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    upper_bound = q3 + (m * iqr)
    lower_bound = q1 - (m * iqr)
    
    return upper_bound, lower_bound


def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def clean_df(df):
    # remove weird column
    df = remove_columns(df, ["Unnamed: 0"])
    # handle outliers
    annincUP, annincLOW = outlier(df, 'logerror')
    # Filter dataframe using multiple conditions

    df = df[(df.logerror < annincUP) & (df.logerror > annincLOW)]
    df = handle_missing_values(df)
    df["fips"] = df.fips.map({6037: "LA", 6059: "Orange", 6111: "Ventura"})
    dummy_df = pd.get_dummies(df[["fips"]], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(columns=['id','parcelid', 'buildingqualitytypeid','heatingorsystemtypeid', 'propertylandusetypeid', 'propertylandusetypeid'])

    return df 

def split_data(df):
    '''
    Takes in two arguments the dataframe name and the ("stratify_name" - must be in string format) to stratify  and 
    return train, validate, test subset dataframes will output train, validate, and test in that order
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123)
    train, validate = train_test_split(train, #second split
                                    test_size=.25, 
                                    random_state=123)
    return train, validate, test

def rename_col(df, list_of_columns=[]): 
    '''
    Take df with incorrect names and will return a renamed df using the 'list_of_columns' which will contain a list of appropriate names for the columns  
    '''
    df = df.rename(columns=dict(zip(df.columns, list_of_columns)))
    return df

def mm_scale(x_train, x_validate, x_test):
    """
    Apply MinMax scaling to the input data.

    Args:
        x_train (pd.DataFrame): Training data features.
        x_validate (pd.DataFrame): Validation data features.
        x_test (pd.DataFrame): Test data features.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Scaled versions of the input data
            (x_train_scaled, x_validate_scaled, x_test_scaled).
    """
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(x_train)


    x_train_scaled = scaler.transform(x_train)
    x_validate_scaled = scaler.transform(x_validate)
    x_test_scaled = scaler.transform(x_test)

    col_name = list(x_train.columns)

    x_train_scaled, x_validate_scaled, x_test_scaled = pd.DataFrame(x_train_scaled), pd.DataFrame(x_validate_scaled), pd.DataFrame(x_test_scaled)
    x_train_scaled, x_validate_scaled, x_test_scaled  = rename_col(x_train_scaled, col_name), rename_col(x_validate_scaled, col_name), rename_col(x_test_scaled, col_name)
    
    return x_train_scaled, x_validate_scaled, x_test_scaled