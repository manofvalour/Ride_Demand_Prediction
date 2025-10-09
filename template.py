import os
import logging
from pathlib import Path

logging.basicConfig(level= logging.INFO, format = "[%(asctime)s]: %(levelname)s: %(message)s")

project_name = "RideDemandForcast"

list_of_dir = [
    '.github/workflows/.gitkeep',
    'config/config.yaml',
    'params.yaml',
    'schema.yaml',
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/component/__init__.py',
    f'src/{project_name}/entity/__init__.py',
    f'src/{project_name}/entity/config_entity.py',
    f'src/{project_name}/config/__init__.py',
    f'src/{project_name}/config/configuration.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/utils/common_utils.py',
    f'src/{project_name}/pipeline/__init__.py',
    f'src/{project_name}/pipeline/training_pipeline.py',
    f'src/{project_name}/pipeline/prediction_pipeline.py',
    f'src/{project_name}/logger/__init__.py',
    f'src/{project_name}/logger/logger.py',
    f'src/{project_name}/exception/__init__.py',
    f'src/{project_name}/exception/customexception.py',
    f'src/{project_name}/constants/__init__.py',
    'notebook/research.ipynb',
    'templates/index.html',
    'static/style.css',
    'DockerFile'

]


for dir in list_of_dir:
    file_path = Path(dir)
    file_dir, filename = os.path.split(file_path)

    if file_dir!="":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"directory created successfully {file_dir}")

    if not (os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path, 'w') as file:
            pass
    
    else:
        logging.info('file exists already')
