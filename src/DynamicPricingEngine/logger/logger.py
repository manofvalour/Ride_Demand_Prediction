"""Simple logger configuration for the DynamicPricingEngine package.

Creates a timestamped log file under `logs/` and exposes a module
level `logger` configured to write to both file and stdout.
"""

import os
import sys
from datetime import datetime
import logging

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# creating the log directory
log_dir = os.path.join(os.getcwd(), 'logs', LOG_FILE_NAME)
os.makedirs(log_dir, exist_ok=True)

file_name = os.path.join(log_dir, LOG_FILE_NAME)

logging.basicConfig(
    format="[ %(asctime)s ]: %(name)s: %(levelname)s: %(module)s: %(lineno)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(file_name),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('DynamicPricingEngine')