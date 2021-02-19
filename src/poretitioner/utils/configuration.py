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
import toml
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from poretitioner.utils.core import PathLikeOrString


@dataclass(frozen=True)
class CONFIG:
    GENERAL = "general"
    SEGMENTATION = "segmentation"
    FILTER = "filter"


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

    def validate(self):
        # assert self.n_workers > 0
        # assert self.captures_per_f5 > 0
        pass

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
class SegmentConfiguration(BaseConfiguration):
    bulkfast5: str
    n_captures_per_file: int
    voltage_threshold: int
    signal_threshold_frac: float
    translocation_delay: float
    terminal_capture_only: bool
    delay: int
    end_tolerance: float
    good_channels: List[int]
    open_channel_prior_mean: int
    open_channel_prior_stdv: int
    alt_open_channel_pA: int
    terminal_capture_only: bool
    # TODO: Pipe through filtering https://github.com/uwmisl/poretitioner/issues/43 https://github.com/uwmisl/poretitioner/issues/68
    filters: Dict
    delay: int

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
        command_line_args = { arg: value for arg, value in command_line_args.items() if hasattr(self, arg)}
        config = config if config is not None else {}
        # Command line args take precidence over configuration files in the event of a conflict.
        combined = {**config, **command_line_args}
        for configuration_key, value in combined.items():
            object.__setattr__(self, configuration_key, value)

    def validate(self):
        raise NotImplementedError("Not implemented configuration")


@dataclass(frozen=True)
class FilterConfig:
    name: str
    attributes: Dict[str, Any]
    filepath: Optional[str]


@dataclass(frozen=True)
class FilterConfiguration(BaseConfiguration):
    """Configuration for the filtering step.
    """

    """Whether to only consider captures that were in the pore until the voltage was reversed.
    """
    only_use_ejected: bool
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


@dataclass(frozen=True)
class ClassifierConfiguration(BaseConfiguration):
    classifier: str
    version: str
    start_obs: int
    end_obs: int
    min_confidence: float

    def validate(self):
        raise NotImplementedError("Not implemented configuration")


def readconfig(path, command_line_args=None):
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

    read_config = toml.load(config_path)
    #config = ConfigParser()

    gen_config = read_config[CONFIG.GENERAL]
    seg_config = read_config[CONFIG.SEGMENTATION]

    #config.read(config_path)
    #config = config

    segmentation_configuration = SegmentConfiguration(config=seg_config, command_line_args=command_line_args)
    general_configuration = GeneralConfiguration(config=gen_config, command_line_args=command_line_args)

    configs = {
        CONFIG.GENERAL: general_configuration,
        CONFIG.SEGMENTATION: segmentation_configuration
    }

    return configs

    # TODO: Return configuration https://github.com/uwmisl/poretitioner/issues/73


def read_segmentation(config, command_line_args=None):
    seg_config = readconfig(config, command_line_args=command_line_args)
    print(seg_config)
