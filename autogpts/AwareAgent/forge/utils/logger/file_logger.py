import logging
import os

from forge.utils.logger.console_logger import ForgeLogger
from forge.sdk.config.storage import get_permanent_storage_path


class FileLogger(logging.Logger):
    _instances = {}  # A class-level attribute used to store unique instances

    def __new__(cls, name, *args, **kwargs):
        # If an instance with this name exists, return it
        if name in cls._instances:
            return cls._instances[name]

        # Create a new instance because one doesn't exist
        instance = super(FileLogger, cls).__new__(cls)
        cls._instances[name] = instance
        return instance

    def __init__(self, name, level=logging.NOTSET):
        # If we have already initialized this instance, we don't want to do it again
        if getattr(self, '_initialized', False):
            return

        super().__init__(name, level)

        # File handler setup
        logger_path = os.path.join(get_permanent_storage_path(), f"{name}.log")
        os.makedirs(os.path.dirname(logger_path), exist_ok=True)
        file_handler = logging.FileHandler(logger_path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.addHandler(file_handler)

        # Setup ForgeLogger as the console logger
        self.console_logger = ForgeLogger(name)

        self._initialized = True

    def info(self, msg, *args, should_print=True, **kwargs):
        super().info(msg, *args, **kwargs)  # This logs to the file
        if should_print:
            self.console_logger.info(msg, *args, **kwargs)  # This logs to the console

    def debug(self, msg, *args, should_print=True, **kwargs):
        super().debug(msg, *args, **kwargs)
        if should_print:
            self.console_logger.debug(msg, *args, **kwargs)

    def warning(self, msg, *args, should_print=True, **kwargs):
        super().warning(msg, *args, **kwargs)
        if should_print:
            self.console_logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, should_print=True, **kwargs):
        super().error(msg, *args, **kwargs)
        if should_print:
            self.console_logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, should_print=True, **kwargs):
        super().critical(msg, *args, **kwargs)
        if should_print:
            self.console_logger.critical(msg, *args, **kwargs)
