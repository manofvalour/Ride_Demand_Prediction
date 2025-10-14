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



