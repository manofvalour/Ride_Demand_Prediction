import sys
from src.DynamicPricingEngine.logger.logger import logger


class RideDemandException(Exception):
    def __init__(self, error_detail, error:sys):
        super().__init__(error_detail)

        _,_, exc_tb = error.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno

        self.error_message = f"There is an error in {file_name} at line {line_no} with {error_detail}"

    def __str__(self):
        return self.error_message
