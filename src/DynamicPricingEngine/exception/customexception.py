"""Custom exceptions used across the Ride Demand project.

Defines `RideDemandException` which wraps original exceptions and
annotates them with the source filename and line number for easier
debugging and logging.
"""

import sys


class RideDemandException(Exception):
    """Exception wrapper that records the originating file and line.

    Args:
        error_detail: Original exception or message.
        error (module): The `sys` module instance passed so that
            `error.exc_info()` can be used to extract traceback details.
    """
    def __init__(self, error_detail, error: sys):
        super().__init__(error_detail)

        _, _, exc_tb = error.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_no = exc_tb.tb_lineno

        self.error_message = f"There is an error in {file_name} at line {line_no} with {error_detail}"

    def __str__(self):
        return self.error_message
