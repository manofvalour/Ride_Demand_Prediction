"""Typed dataclasses for configuration objects used across pipelines.

Each dataclass represents the configuration required for a pipeline
stage (ingestion, transformation, training, inference).
"""

from dataclasses import dataclass
from pathlib import Path


################################# DATA INGESTION ####################################
@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion paths and external URLs."""
    root_dir: Path
    taxi_data_url: Path
    weather_data_url: Path
    taxi_data_local_file_path: Path
    weather_data_local_file_path: Path

@dataclass
class DataTransformationConfig:
    """Configuration for feature engineering and shapefile locations."""
    root_dir: Path
    feature_store_url_path: Path
    shapefile_dir: Path
    taxi_zone_shapefile_url: Path
    transformed_data_file_path: Path
    taxi_data_local_file_path: Path
    weather_data_local_file_path: Path

@dataclass
class DataValidationConfig:
    """Placeholder configuration for data validation steps."""
    pass


@dataclass
class ModelTrainerConfig:
    """Configuration for training: paths, splits and Optuna spaces."""
    root_dir: Path
    train_data_path: Path
    val_data_path: Path
    test_data_path: Path
    trained_model_path: Path
    train_split_ratio: float
    target_col: str
    val_split_ratio: float
    optuna_param_spaces: dict

################################# INFERENCE CONFIGURATION ####################################
@dataclass
class InferenceConfig:
    """Configuration used during inference: shapefiles, model/location paths."""
    root_dir: Path
    input_data_path: Path
    model_path: Path
    predictions_output_path: Path
    weather_data_url: str
    shapefile_dir: Path
    taxi_zone_shapefile_url: Path
    zone_lookup_table_url: Path
    geojson_output_path: Path







    




