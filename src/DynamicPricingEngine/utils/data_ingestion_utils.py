import os,sys
from ensure import ensure_annotations
from pathlib import Path
from typing import Any, List
from datetime import datetime

from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.logger.logger import logger


@ensure_annotations
def leap_year(year:int)->bool:

    try:

        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return True
        else:
            return False
    
    except Exception as e:
        raise RideDemandException(e,sys)

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
