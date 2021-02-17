"""
========================
configuration.py
========================

In computational biology and bioinformatives, it's common practice to prefer
configuration files to command line arguments.
This module is responsible for parsing the application's configuration file.

Still working on the configuration step.

"""
import math
from abc import ABCMeta, abstractmethod
from configparser import ConfigParser
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, Union

import numpy as np

from .core import PathLikeOrString


def get_absolute_path(path: PathLikeOrString) -> Path:
    """Gets the absolute path for a file path.
    This method takes care of things like substituting "~" for the home directory.

    Parameters
    ----------
    path : str or PathLike
        Path to a file.

    Returns
    -------
    Path
        Absolute path to the file.
    """
    absolute_path = Path(path).expanduser().resolve()
    return absolute_path


def is_valid_directory(path: PathLikeOrString) -> bool:
    """Whether a path represents a valid directory.

    Parameters
    ----------
    path : str or PathLike
        A path to a directory.

    Returns
    -------
    bool
        Whether this path refers to an existing directory.
    """
    is_directory = get_absolute_path(path).is_dir()
    return is_directory


def number(num_str: str) -> float:
    """[summary]

    Parameters
    ----------
    num_str : str
        [description]

    Returns
    -------
    float
        [description]
    """
    if num_str == "∞" or num_str == "math.inf" or num_str == "inf":
        return math.inf
    elif num_str == "-∞" or num_str == "-math.inf" or num_str == "-inf":
        return -math.inf
    return float(num_str)


def str_to_bool(string: str) -> bool:
    """Takes a string, converts it to a boolean. Ignores whitespace.

    Parameters
    ----------
    string : str
        Boolean-like string (e.g. "True", "true", "   true", " TRUE  " all return true).

    Returns
    -------
    bool
        The bool equivalent of the string.

    Raises
    ------
    ValueError
        String could not be converted to a boolean.
    """
    string = string.strip()
    if string.upper() == "TRUE":
        return True
    elif string.upper() == "FALSE":
        return False
    raise ValueError(
        f"'{string}' is not parsible into 'True' or 'False'. Double-check that it's spelled correctly."
    )


class BaseConfiguration(metaclass=ABCMeta):
    """Abstract base class for configurations. This class isn't meant to be instantiated directly.
    All configuration classes should include a `validate` method, which will throw an exception
    for invalid data.
    """

    @abstractmethod
    def validate(self) -> bool:
        """Return true if and only if this instance represents a valid configuration.
        Throws a ValueError exception otherwise

        Returns
        -------
        bool
            True if and only if this instance was a valid configuration.
        """
        raise ValueError("Configuration was invalid.")


@dataclass(frozen=True)
class GeneralConfiguration(BaseConfiguration):
    n_workers: int
    capture_directory: str
    captures_per_f5: int

    def validate(self):
        # assert self.n_workers > 0
        # assert self.captures_per_f5 > 0
        pass


@dataclass(frozen=True)
class SegmentConfiguration(BaseConfiguration):
    bulkfast5: str
    segmentedfast5: str
    voltage_threshold: float
    signal_threshold_frac: float
    translocation_delay: float
    alt_open_channel_pA: float
    terminal_capture_only: bool
    delay: int
    end_tol: float
    signal_threshold_frac: float
    alt_open_channel_pA: int
    terminal_capture_only: bool
    # TODO: Pipe through filtering https://github.com/uwmisl/poretitioner/issues/43 https://github.com/uwmisl/poretitioner/issues/68
    filters: Dict
    delay: int
    end_tol: int

    def __init__(self, command_line_args: Dict = None, config: Dict = None) -> None:
        """[summary]

        Parameters
        ----------
        command_line_args : Dict, optional
            [description], by default None
        config : Dict, optional
            [description], by default None
        """
        command_line_args = command_line_args if command_line_args is not None else {}
        config = config if config is not None else {}
        # Command line args take precidence over configuration files in the event of a conflict.
        combined = {**config, **command_line_args}
        for configuration_key, value in combined.items():
            object.__setattr__(self, configuration_key, value)

    def validate(self):
        raise NotImplementedError("Not implemented configuration")


@dataclass(frozen=True)
class FilterConfiguration(BaseConfiguration):
    # Filters to apply to the data.
    # Valid filters take the form:
    # "name": "value"
    # Whether to filter based on the signal mean.
    use_mean: bool = True
    # Minimum signal mean to pass. Defaults to -∞ if not present.
    mean_min: float = -math.inf
    # Maximum signal mean to pass. Defaults to +∞ if not present.
    mean_max: float = math.inf

    # Whether to filter based on the signal standard deviation.
    use_stdv: bool = True
    # Minimum signal standard deviation to pass. Defaults to -∞ if not present.
    stdv_min: float = -math.inf
    # Maximum signal standard deviation to pass. Defaults to +∞ if not present.
    stdv_max: float = math.inf

    # Whether to filter based on the signal median.
    use_median: bool = True
    # Minimum signal median to pass. Defaults to -∞ if not present.
    median_min: float = -math.inf
    # Maximum signal median to pass. Defaults to +∞ if not present.
    median_max: float = math.inf

    # Whether to filter based on the minimal signal value.
    use_minimum: bool = True
    # Minimum signal absolute value to pass. Defaults to -∞ if not present.
    minimum: float = -math.inf

    # Whether to filter based on the maximal signal value.
    use_max: bool = True
    # Maximum signal absolute value to pass. Defaults to +∞ if not present.
    maximum: float = math.inf

    # Whether to filter based on the signal length.
    use_length: bool = False
    # What minimal signal length to allow.Defaults to -∞ if not present.
    length_min: float = -math.inf
    # The maximal signal length to allow. Defaults to +∞ if not present.
    length_max: float = math.inf

    def validate(self):
        raise NotImplementedError("Not implemented")


@dataclass(frozen=True)
class QuantifierConfiguration(BaseConfiguration):
    def validate(self):
        raise NotImplementedError("Not implemented configuration")


# TODO: Katie Q: Is there ever a circumstance where we'll want to use more than one classifier in the same run?
@dataclass(frozen=True)
class ClassifierConfiguration(BaseConfiguration):
    classifier: str
    version: str
    start_obs: int
    end_obs: int
    min_confidence: float

    def validate(self):
        raise NotImplementedError("Not implemented configuration")


def readconfig(path):
    """Read configuration from the path.

    Exceptions
    ----------
    FileNotFoundError: If path could not be resolved
    Parameters
    ----------
    path : Pathlike
        Path to the Poretitioner configuration file.
    """
    config_path = get_absolute_path(path)
    config = ConfigParser()

    config.read(config_path)
    config = config

    # TODO: Return configuration https://github.com/uwmisl/poretitioner/issues/73


def read_segmentation(config):
    readconfig(config)
