import numpy as np
import pandas as pd
import os

# a function that returns the key used in the matching process
def get_key(key):
    x = str(key[0])
    for i in range(1, len(key)):
        x += str(key[i])
    return x

# a function that builds the key 
def cat_key(x):
    if pd.isna(x) :
        return ''
    else:
        return x

# a function that applies a mask to the variables used to construct the key
def get_cat_key(df):
    return np.array(df.applymap(cat_key))

# the function returns all the keys calculated above
def get_all_keys(df, _list):
    keys_ = get_cat_key(df[_list])

    all_keys = []

    for i in range(keys_.shape[0]):
        all_keys.append(get_key(keys_[i, :]))

    return all_keys

# a function returning all previously calculated keys and codes 
def get_keys(df, _list):
    return get_all_keys(df, _list)

# auxiliary function for console cleaning 
def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If the machine is running on Windows, use cls
        command = 'cls'
    os.system(command)

# auxiliary function to rename columns after each match 
def rename_cols(df):
    '''
    Operates on a dataframe resulting from a join.
    Identifying the cases in which there was a renaming of similar columns
    with different information, consolidating them.

    params:
        df (Dataframe) : The dataframe on which you want to operate

    returns:
        df (Dataframe) : The same df dataframe with the consolidated columns

    example:
        df_1 = df_1.merge(df_2, how = 'left')
        df_1 = rename_cols(df_1)
        >>
    '''
    cols = []
    for i in df.columns:
        cols.append(i.replace('_x', ''))
        cols.append(i.replace('_y', ''))
   
    cols = [*set(cols)]
    
    for i in cols:
        try:
            df[i+'_x'] = df[i+'_x'].fillna(df[i+'_y'])
            df = df.drop(columns=[i+'_y'])
            df.rename(columns = {i+'_x':i}, inplace = True)
        except:
            None

    return df

####################################################################################

def like_filter(df, filters):
    mask = []
    for i in filters:
        try:
            mask.append(df.filter(like = i).columns[0])
        except:
            None
    return mask

def make_match(df1, df2, subset, key, dropna_filters):
    df1 = df1.drop_duplicates(subset = key)
    df3 = df2[df2[dropna_filters].isnull().any(axis=1)].copy()
    df3 = df3.merge(df1, how='left', on=key)
    df3 = rename_cols(df3)
    if len(df3) == len(df2):
        df2 = df2.combine_first(df3)
    else:
        df2 = df2.merge(df1, how='left', on=key)
        df2 = rename_cols(df2)
    del df3

    if subset != None:
        df2 = df2.drop_duplicates(subset=subset)
    else:
        df2 = df2.drop_duplicates()
    return df2

def return_df(list_df_):
    df_ = pd.DataFrame()
    for df in list_df_:
        df = rename_cols(df)
        df_ = pd.concat([df, df_], ignore_index=True)
    return df_

# a main function that performs the piecewise (chunks) matching process 
def partial_merge(df1, df2, keys_to, n = None, dropna_filters = [], subset = None):
    if n != None:
        list_df_ = [df1[i:i+n] for i in range(0, df1.shape[0], n)]
        count = 0
        for j in keys_to:
            k = 0
            progress = 0
            for df in list_df_:
                df['key']= get_keys(df, j)
                df2['key'] = get_keys(df2, j)
                print('Progress : ', "{:.2%}".format(count/len(keys_to)))
                print('aggregation : ', "{:,}".format(count))
                print('Total Size df1: ', "{:,}".format(len(df1)), '| Total Size df2: ', "{:,}".format(len(df2)))
                print('Partial Progress : ', "{:.2%}".format(progress/len(list_df_)))
                list_df_[k] = make_match(df2, df, subset, ['key'], dropna_filters)
                clearConsole()
                progress += 1
                k += 1
            count += 1
            if (count >= 1):
                df1 = return_df(list_df_)
    else:
        count = 0
        for j in keys_to:
            df1['key']= get_keys(df1, j)
            df2['key'] = get_keys(df2, j)
            print('Progress : ', "{:.2%}".format(count/len(keys_to)))
            print('aggregation : ', "{:,}".format(count))
            print('Total Size df1: ', "{:,}".format(len(df1)), '| Total Size df2: ', "{:,}".format(len(df2)))
            df1 = make_match(df2, df1, subset, ['key'], dropna_filters)
            clearConsole()
            count += 1
    df1 = df1.drop(columns=['key'])
    return df1
####################################################################################