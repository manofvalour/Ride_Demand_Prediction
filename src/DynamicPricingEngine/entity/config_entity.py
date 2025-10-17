from dataclasses import dataclass
from pathlib import Path


################################# DATA INGESTION ####################################
@dataclass
class DataIngestionConfig:
    root_dir: Path
    taxi_data_url: Path
    weather_data_url: Path
    taxi_data_local_file_path: Path
    weather_data_local_file_path: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    feature_store_url_path: Path
    shapefile_dir: Path
    taxi_zone_shapefile_url: Path
    transformed_data_file_path: Path

@dataclass
class DataValidationConfig:
    pass


@dataclass
class ModelTrainerConfig:
    pass




