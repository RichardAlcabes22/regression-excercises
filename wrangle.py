# inside acquire.py script:
from env import uname, pwd, host
import env
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

######### USE THIS FOR THE zillow DATASET!!!!
def get_df():
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename,index_col=False)
    else:
        sql_query = """
                SELECT  bedroomcnt as beds,
                    bathroomcnt as baths,
                    calculatedfinishedsquarefeet as sqft,
                    taxvaluedollarcnt as taxable_value,
                    yearbuilt as built,
                    taxamount as tax,
                    fips,
                    propertylandusetypeid
                FROM properties_2017
                WHERE propertylandusetypeid = 261
                """
    
        # Read in DataFrame from Codeup db.
        df = pd.read_sql(sql_query, env.get_db_url('zillow'))
        df.to_csv(filename,index=False)
        return df
    
def clean_zillow(df):
    '''
    clean_szillow will take in df and will remove propertyland..., rows with NULL values in any cell and will
    cast floats into int.
    
    args: df
    return: df (clean)
    
    '''
    df = df.drop(columns='propertylandusetypeid')
    df = df.dropna()
    df = df.astype(int)
    return df

def split_zillow(df):
    train_val,test = train_test_split(df,
                                     random_state=2013,
                                     train_size=0.7)
    train, validate = train_test_split(train_val,
                                      random_state=2013,
                                      train_size=0.8)
    return train, validate, test

################ FINAL SUMMARY FUNCTION ###############

def wrangle_zillow():
    '''
    This function reads the zillow data from the Codeup db into a df and cleans the data as follows:
    Drop the property land use id
    Drop rows with missingness
    Convert all floats to INT

    BE SURE TO create a three-part variable to capture output of wrangle_zillow....ie train,validate,test = wrangle_zillow()
    '''    
    train, validate, test = split_zillow(
            clean_zillow(
                get_df()))
    return train, validate, test


# SCALE  continuous X_col with QUANTILE TRANSFORMER with 'normal' DIST

def scale_zillow_col(train,validate,test):
    '''
    
    accepts three dfs (train,val,test) and constructs X and y_train, _val, _test.  Then applies a 
    QuantileTransformer with 'normal' output to the specified column(s).
    Outputs normal, standard arrays based upon the scaled values of each col.
    
    '''
    X_train = train.drop(columns=['beds','baths','taxable_value','built','fips'])
    y_train = train['taxable_value']
    y_train = pd.DataFrame(y_train)

    X_validate = validate.drop(columns=['beds','baths','taxable_value','built','fips'])
    y_validate = validate['taxable_value']
    y_validate = pd.DataFrame(y_validate)

    X_test = test.drop(columns=['beds','baths','taxable_value','built','fips'])
    y_test = test['taxable_value']
    y_test = pd.DataFrame(y_test)
    
    scaler = sklearn.preprocessing.QuantileTransformer(output_distribution='normal')
    scaler.fit(X_train)

    train_col_scaled = scaler.transform(X_train)
    validate_col_scaled = scaler.transform(X_validate)
    test_col_scaled = scaler.transform(X_test)

    return train_col_scaled,validate_col_scaled,test_col_scaled

def remove_outliers(df, col_list, k=1.5):
    '''
    remove outliers from a dataframe based on a list of columns
    using the tukey method.
    returns a single dataframe with outliers removed
    '''
    col_qs = {}
    for col in col_list:
        col_qs[col] = q1, q3 = df[col].quantile([0.25, 0.75])
    for col in col_list:
        iqr = col_qs[col][0.75] - col_qs[col][0.25]
        lower_fence = col_qs[col][0.25] - (k*iqr)
        upper_fence = col_qs[col][0.75] + (k*iqr)
        print(type(lower_fence))
        print(lower_fence)
        df = df[(df[col] > lower_fence) & (df[col] < upper_fence)]
    return df