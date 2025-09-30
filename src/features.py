import numpy as np
import pandas as pd

# Import function directly (more controlled than import *)
from src import *

def cast_datatypes(df, cast_type, numeric_type=None, date_format=None, c_time_zone=None, c_include=None, c_exclude=None):
    
    if c_exclude is None:
        c_exclude = []

    if c_include is None:
        available_columns = [col for col in df.columns if col not in c_exclude]
    else:
        available_columns = [col for col in c_include if col not in c_exclude]

    if cast_type == 'string':
        for column in available_columns:
            df[column] = df[column].astype("string")
    
    elif cast_type == 'numeric':
        df = convert_object_to_numeric(df, type=numeric_type, include=c_include, exclude=c_exclude)
    
    elif cast_type == 'category':
        for column in available_columns:
            df[column] = df[column].astype("category")
    
    elif cast_type == 'boolean':
        for column in available_columns:
            if df[column].dropna().isin([0, 1]).all():
                df[column] = df[column].astype("boolean")
            else:
                df[column] = df[column].astype("Int64").astype("boolean")
    
    elif cast_type == 'datetime':
        tz = c_time_zone if c_time_zone is not None else ""
        df = normalize_datetime(df, include=c_include, exclude=c_exclude, frmt=date_format, time_zone=tz)

    else:
        raise ValueError(f"Unsupported cast_type: {cast_type}")
    
    return df
    