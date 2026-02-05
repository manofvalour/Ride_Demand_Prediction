"""Small utilities used during data ingestion and memory optimisation.

Contains helpers for calculating month lengths, downcasting DataFrame
dtypes for memory efficiency and other ingestion-time helpers.
"""

import sys
from ensure import ensure_annotations
from datetime import datetime
import pandas as pd
import gc

from src.DynamicPricingEngine.exception.customexception import RideDemandException


@ensure_annotations
def leap_year(year: int) -> bool:
    """Return True if `year` is a leap year according to Gregorian rules."""
    try:
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    except Exception as e:
        raise RideDemandException(e, sys)

@ensure_annotations
def time_subtract(date:str)->int:
    """
    A function to calculate the number of days to subract from the data ingesting
    Args: 
        date (str): the date to use
    Return:
        days(int): the number of days calculated
       """
    try:

        date = datetime.strptime(date, "%Y-%m-%d")
        days = 0

        if (date.month != 2) and (date.month not in (4,9,6,11)):
            days=31

        elif (date.month != 2) and (date.month in (4,9,6,11)):
            days=30

        elif (date.month == 2) and (leap_year(date.year)):
            days=29

        else:
            days=28

        return days
    
    except Exception as e:
        raise RideDemandException(e,sys)
    
@ensure_annotations
def dtype_downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric and object columns to reduce DataFrame memory.

    Numeric columns are downcast to the smallest appropriate integer
    or float subtype. Object columns with low cardinality are converted
    to `category`.

    Returns:
        pd.DataFrame: The same dataframe with optimized dtypes.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        # Handle Integers and Floats
        if pd.api.types.is_numeric_dtype(col_type):
            if pd.api.types.is_integer_dtype(col_type):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')

        # Handle Objects (Strings) -> Categorical
        elif pd.api.types.is_object_dtype(col_type):
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:  # If < 50% are unique
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")
    gc.collect()
    return df
