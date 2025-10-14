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
def save_pickle(file:object, save_path:str)-> None:
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