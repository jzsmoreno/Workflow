import os

import numpy as np
import pandas as pd
from IPython.display import clear_output


# Optimized function to build the key
def get_key(key):
    return "".join(map(str, key))


# Optimized function to handle NaN efficiently
def cat_key(x):
    return "" if pd.isna(x) else str(x)


# Optimized function to apply mask and construct keys
def get_cat_key(df):
    return df.apply(lambda col: col.map(cat_key)).values


# Optimized function to return all keys
def get_all_keys(df, _list):
    keys_ = get_cat_key(df[_list])
    return ["".join(map(str, row)) for row in keys_]


# Optimized function to get all keys
def get_keys(df, _list):
    return get_all_keys(df, _list)


# Auxiliary function to clear console
def clear_console():
    clear_output(wait=True)
    os.system("cls" if os.name in ("nt", "dos") else "clear")


# Optimized function to rename columns after matching
def rename_cols(df):
    for col in df.columns:
        if "_x" in col:
            base_col = col.replace("_x", "")
            if base_col + "_y" in df.columns:
                df[base_col] = df[base_col + "_x"].fillna(df[base_col + "_y"])
                df.drop([base_col + "_x", base_col + "_y"], axis=1, inplace=True)
                df.rename(columns={base_col + "_x": base_col}, inplace=True)
    return df


####################################################################################


# Optimized like filter function
def like_filter(df, filters):
    return [col for i in filters for col in df.filter(like=i).columns]


# Optimized match-making function with condition for dropping NaN filters
def make_match(df1, df2, subset, key, dropna_filters):
    df1 = df1.drop_duplicates(subset=key)
    df3 = df2[df2[dropna_filters].isnull().any(axis=1)].merge(df1, how="left", on=key)
    df3 = rename_cols(df3)

    if len(df3) == len(df2):
        df2 = df2.combine_first(df3)
    else:
        df2 = df2.merge(df1, how="left", on=key)
        df2 = rename_cols(df2)

    if subset:
        df2 = df2.drop_duplicates(subset=subset)
    else:
        df2 = df2.drop_duplicates()

    return df2


# Optimized function to return concatenated dataframe
def return_df(list_df_):
    return pd.concat(list_df_, ignore_index=True)


# Optimized partial merge function with reduced print overhead
def partial_merge(df1, df2, keys_to, n=None, dropna_filters=[], subset=None):
    if n:
        list_df_ = [df1[i : i + n] for i in range(0, df1.shape[0], n)]
        count = 0
        for j in keys_to:
            progress = 0
            for k, df in enumerate(list_df_):
                df["key"] = get_keys(df, j)
                df2["key"] = get_keys(df2, j)
                print("Progress : ", "{:.2%}".format(count / len(keys_to)))
                print("aggregation : ", "{:,}".format(count))
                print(
                    "Total Size df1: ",
                    "{:,}".format(len(df1)),
                    "| Total Size df2: ",
                    "{:,}".format(len(df2)),
                )
                print("Partial Progress : ", "{:.2%}".format(progress / len(list_df_)))
                list_df_[k] = make_match(df2, df, subset, ["key"], dropna_filters)
                clear_console()
                progress += 1
                k += 1
            count += 1
            if count >= 1:
                df1 = return_df(list_df_)
    else:
        count = 0
        for j in keys_to:
            df1["key"] = get_keys(df1, j)
            df2["key"] = get_keys(df2, j)
            print("Progress : ", "{:.2%}".format(count / len(keys_to)))
            print("aggregation : ", "{:,}".format(count))
            print(
                "Total Size df1: ",
                "{:,}".format(len(df1)),
                "| Total Size df2: ",
                "{:,}".format(len(df2)),
            )
            df1 = make_match(df2, df1, subset, ["key"], dropna_filters)
            clear_console()
            count += 1
    df1 = df1.drop(columns=["key"])
    return df1


####################################################################################
