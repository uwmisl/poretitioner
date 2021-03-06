"""
==========
logger.py
==========

This module centralizes the application's logging logic.
It has methods for configuring the logger verbosity and formatting.
Most importantly, it provides a method for accessing a pre-configured logger.

"""
import logging
from logging import Logger
from typing import *  # I know people don't like import *, but I think it has benefits for types (doesn't impede people from being generous with typing)

import colorlog

Logger = NewType("Logger", Logger)
__all__ = ["configure_root_logger", "getLogger", "verbosity_to_log_level", "Logger"]


def verbosity_to_log_level(verbosity: int = 0) -> int:
    """Converts a verbosity to a logging level from the standard library logging module.

    0 - Log errors only.

    1 - Additionally log warnings.

    2 - Additionally log general information.

    3 - Additionally log debug information.

    Parameters
    ----------
    verbosity : int, optional
        The verbosity level, by default 0
    Returns
    -------
    int
        A logging level that can be used by the standard library logging module.
    """

    verbosity = max(0, verbosity)
    log_level = logging.WARNING
    if verbosity == 0:
        log_level = logging.ERROR
    elif verbosity == 1:
        log_level = logging.WARNING
    elif verbosity == 2:
        log_level = logging.INFO
    else:
        log_level = logging.DEBUG
    return log_level


def configure_root_logger(verbosity: int = 0, debug: bool = False, format: Optional[str] = None):
    """Configures a logger for usage throughout the application.

    This should be called once during the initialization process, configured
    based on the user's desired logging verbosity.

    Parameters
    ----------
    verbosity : int, optional
        How verbose the logging should be on a scale of 0 to 3. Anything higher is treated as a 3.
    debug : bool, optional
        Whether to append filename and line number to the log, by default False
    FORMAT : str, optional
        Formatting string to use, by default None. https://docs.python.org/3/library/logging.html#logging.Formatter
    """
    log_level = verbosity_to_log_level(verbosity=verbosity)

    root_logger = colorlog.getLogger(__name__)
    handler = colorlog.StreamHandler()

    # DEFAULT_FORMAT Adds filename and line number for debug-mode outputs
    DEFAULT_FORMAT = f"%(log_color)s[%(asctime)s] [%(levelname)-8s] --- %(message)s {'(%(filename)s:%(lineno)s)' if debug else '' }"
    format = format if format is not None else DEFAULT_FORMAT
    formatter = colorlog.ColoredFormatter(format, style="%")
    handler.setFormatter(formatter)

    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


def getLogger() -> Logger:
    """Gets the application logger.

    Use this method the same way you'd use the Python standard library logging module.
    (e.g. getLogger().warning("This is a warning")).

    Make sure the root logger is configured via `configure_root_logger` before using this method.

    Returns
    -------
    Logger
        A configured Logger object.
    """
    logger = logging.getLogger(__name__)
    return logger
