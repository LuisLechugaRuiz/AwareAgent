import logging
import os

from forge.utils.logger.console_logger import ForgeLogger
from forge.sdk.config.storage import get_permanent_storage_path


class FileLogger:
    def __init__(self, process_name: str):
        self.file_logger = self.create_logger(process_name)
        self.console_logger = ForgeLogger(__name__)

    def _create_logger(self, logger_name):
        # check if logger with given name already exists
        if logger_name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(logger_name)
        else:
            # create new logger with given name
            logger = logging.getLogger(logger_name)
            logger.propagate = False
            logger.setLevel(logging.INFO)
            os.makedirs(os.path.dirname(logger_name), exist_ok=True)

            # create file handler which logs even debug messages
            fh = logging.FileHandler(f"{logger_name}.log")
            fh.setLevel(logging.INFO)

            # create formatter and add it to the handler
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            fh.setFormatter(formatter)

            # add the handler to the logger
            logger.addHandler(fh)

            # store the file handler
            self.file_handler = fh

        return logger

    def create_logger(self, process_name: str) -> "logging.Logger":
        logger_path = os.path.join(get_permanent_storage_path(), process_name)
        return self._create_logger(logger_path)

    def log(
        self,
        message: str,
        log_level: int = logging.INFO,
        should_print: bool = True,
    ):
        if should_print:
            self.file_logger.log(level=log_level, msg=str(message))
            self.console_logger.log(level=log_level, msg=str(message))
