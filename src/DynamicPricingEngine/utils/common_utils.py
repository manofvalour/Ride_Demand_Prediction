import yaml
from box import ConfigBox
import dill
import os,sys
from ensure import ensure_annotations
from typing import Any, List, Union
from types import NoneType
from box.exceptions import BoxValueError
from datetime import datetime, timedelta
from pathlib import Path
import requests
import zipfile
import io
import geopandas as gpd
import time
import pandas as pd

from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.logger.logger import logger


### utility tools to build
## save_obj
## load_obj
## create_dir
## read yaml

@ensure_annotations
def read_yaml(path_to_yaml_file: Path)->ConfigBox:
    """

    Function to read yaml file from a given path

    Args:
        path_to_yaml_file (str): path to the yaml file

    Raises:
        ValueError: if yaml file is empty
        e: file is empty

    Returns:
        ConfigBox: ConfigBox type
    """

    try:
        with open(path_to_yaml_file, 'r') as yaml_file:
            file = yaml.safe_load(yaml_file)
            logger.info(f'successfully load the yaml file from path: {path_to_yaml_file}')

            return ConfigBox(file)
        
    except BoxValueError:
        raise ValueError(f"{path_to_yaml_file} doesn't contain yaml file")

    except Exception as e:
        logger.error(f"{path_to_yaml_file} doesn't contain yaml file")
        raise RideDemandException(e, sys)
    

@ensure_annotations
def create_dirs(dir_path: Union[str, List[str]], verbose: bool = True) -> NoneType:
    """
    Create one or more directories.

    Args:
        dir_path (str | List[str]): A single directory path or a list of paths to create.
        verbose (bool, optional): Whether to log directory creation. Defaults to True.

    Returns:
        None
    """
    try:
        # Normalize to a list so we can always iterate
        if isinstance(dir_path, str):
            dir_path = [dir_path]

        for path in dir_path:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"Successfully created directory at: {path}")

    except Exception as e:
        logger.error(f"{e}")
        raise RideDemandException(e, sys)

@ensure_annotations
def create_dir(dir_path:List, verbose:bool=True):
    """
    Function to create directiory

    Args:
        dir_path (List): The list of directory path to be create
        ignore_log (bool, optional): ignore if multiple directory are to be created. Default to True

    Return:
        None
    """

    try:
        for path in dir_path:
            os.makedirs(path, exist_ok=True)

            if verbose:
                logger.info(f"successfully created directory at: {dir_path}")

    except Exception as e:
        logger.error(f"{e}")
        raise RideDemandException(e,sys)


@ensure_annotations
def save_pickle(file:object, save_path)-> None:
    """
    function to save pickle file to a directory

    Args:
        file (object): file to be saved
        save_path (str): the directory path to save the file

    Return:
        None
    """

    try:
        logger.info(f'creating the path to store file:{save_path}' )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as file_path:
            dill.dump(file,file_path)

            logger.info(f"pickle file saved to path: {save_path}")

    except Exception as e:
        raise RideDemandException(e,sys)
    
@ensure_annotations
def load_pickle(file_path:str)->object:
    """
    function to load pickle file

    Args:
        file_path(str): directory to the location of the pickle file
    
    Return:
        binary file (object): the pickled file
    """

    try:
        if not os.path.exists(file_path):
            raise Exception(f'{file_path} path does not exist')
        
        with open(file_path, 'rb') as pickle_file:
            file= dill.load(pickle_file)

            logger.info(f"Pickle file successfully loaded from path: {file_path}")
            return file

    except Exception as e:
        logger.error(f'Unable to load the pickle file from path: {file_path}')
        raise RideDemandException(e,sys)
    
def save_yaml(file_path:str, yaml_file:object)-> None:

    try:
        pass
    except Exception as e:
        raise RideDemandException(e,sys)
    


#@ensure_annotations
def load_shapefile_from_zipfile(url, extract_to)->gpd.GeoDataFrame:
    """
    Downloads a ZIP file from a URL, extracts it, and loads the shapefile.
    
    Parameters:
        url (str): The URL of the ZIP file.
        extract_to (str): Directory to extract files into.
    
    Returns:
        geopandas.GeoDataFrame: The loaded shapefile as a GeoDataFrame.
    """

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'
    }

    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading shapefile (Attempt {attempt+1})...")
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 202 or (response.status_code == 200 and not response.content):
                logger.info("Server returned 202 or empty body. Waiting 10 seconds...")
                time.sleep(10)
                continue
            
            response.raise_for_status()
            content = response.content

            if zipfile.is_zipfile(io.BytesIO(content)):
               with zipfile.ZipFile(io.BytesIO(content)) as z:
                  z.extractall(extract_to)
               logger.info("Shapefile loaded successfully.")
               break 
            else:
                logger.error(f"Error: Content from server is not a valid zip file.")
              
        except Exception as e:
            logger.error(f"Error on attempt {attempt+1}: {e}")
            time.sleep(5)

    # Find the .shp file
    shp_file = None
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.endswith(".shp"):
                shp_file = os.path.join(root, file)
                break

    if shp_file is None:
        raise FileNotFoundError("No .shp file found in the extracted archive.")

    #Load into GeoDataFrame
    return gpd.read_file(shp_file)

def download_csv_from_web(url)-> pd.DataFrame:
    try:

        headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'
                        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading zone_lookup_table (Attempt {attempt+1})...")
                response = requests.get(url, headers=headers, timeout=30)
                
                if response.status_code == 202 or (response.status_code == 200 and not response.content):
                    logger.info("Server returned 202 or empty body. Waiting 10 seconds...")
                    time.sleep(10)
                    continue
                
                response.raise_for_status()
                break 
                
            except Exception as e:
                print(f"Error on attempt {attempt+1}: {e}")
                time.sleep(5)
        
        return pd.read_csv(io.StringIO(response.text))
    
    except Exception as e:
        logger.error("Can't access the content for the url")
        raise RideDemandException(e,sys)


import geopandas as gpd

def convert_shapefile_to_geojson(shapefile_path: str, geojson_path: str) -> None:
    """
    Converts a shapefile to GeoJSON format.
    """
    # Load the shapefile
    shapefile = gpd.read_file(shapefile_path)
    shapefile = shapefile.to_crs(epsg=4326)

    # Save as GeoJSON
    
    shapefile.to_file(geojson_path, driver='GeoJSON')
    logger.info(f"Shapefile converted to GeoJSON and saved at: {geojson_path}")

