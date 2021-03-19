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
import dataclasses
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, NewType

import numpy as np

from ..getargs import ARG
from ..logger import getLogger, Logger

from .core import stripped_by_keys, PathLikeOrString

from .filtering import Filters, FilterSet
from .filtering import FilterConfig, FilterConfigs
from .filtering import get_filters

@dataclass(frozen=True)
class CONFIG:
    GENERAL = "general"
    SEGMENTATION = "segmentation"
    FILTER = "filters"
    CLASSIFICATION = "classification"


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

    @property
    def valid_field_names(self):
        names = {field.name for field in dataclasses.fields(self.__class__)}
        return names

    def initialize_fields(
        self, command_line_args: Dict = None, config: Dict = None, log: Optional[Logger] = None
    ):
        """[summary]
        #TODO
        Parameters
        ----------
        command_line_args : Dict, optional
            [description], by default None
        config : Dict, optional
            [description], by default None
        """

        log = log if log is not None else getLogger()

        command_line_args = command_line_args if command_line_args is not None else {}
        config = config if config is not None else {}
        # Command line args take precidence over configuration files in the event of a conflict.
        combined = {**config, **command_line_args}

        valid_fields = self.valid_field_names
        for field, value in combined.items():
            if field in valid_fields:
                object.__setattr__(self, field, value)
                log.debug(f"{self.__class__.__name__!s}[{field}] = {value}")
            else:
                log.warning(
                    f"'{field}' is not a valid field for configuration {self.__class__.__name__!s}. Ignoring..."
                )

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

PoretitionerConfig = NewType("PoretitionerConfig", Dict[str, Union[BaseConfiguration, FilterSet]])


@dataclass(frozen=True)
class GeneralConfiguration(BaseConfiguration):
    version: str
    n_workers: int
    capture_directory: str

    def validate(self):
        # assert self.n_workers > 0
        # assert self.captures_per_f5 > 0
        return True

    def __init__(
        self,
        command_line_args: Dict = None,
        config: Dict = None,
        log: Logger = getLogger(),
    ) -> None:
        """[summary]

        Parameters
        ----------
        command_line_args : Dict, optional
            [description], by default None
        config : Dict, optional
            [description], by default None
        """
        command_line_args = stripped_by_keys(
            command_line_args, self.valid_field_names
        )  # Only keep filter-related command line args
        self.initialize_fields(command_line_args=command_line_args, config=config)


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
    end_tolerance: float
    open_channel_prior_mean: int
    open_channel_prior_stdv: int
    terminal_capture_only: bool
    capture_criteria: Filters

    def __init__(
        self,
        command_line_args: Dict = None,
        config: Dict = None,
        log: Logger = getLogger(),
    ) -> None:
        """[summary]

        Parameters
        ----------
        command_line_args : Dict, optional
            Command line arguments for filters, by default None. Any keys outside of
            ARG.FILTER are ignored.
        config : Dict, optional
            Segmentation configuration, by default None
        """
        command_line_args = stripped_by_keys(
            command_line_args, self.valid_field_names
        )  # Only keep filter-related command line args
        self.initialize_fields(command_line_args=command_line_args, config=config, log=log)

    def initialize_fields(self, command_line_args: Dict, config: Dict, log: Optional[Logger] = None):
        super().initialize_fields(command_line_args=command_line_args, config=config, log=log)

        # Overwrite with actual capture criteria
        capture_criteria_filter_configs: FilterConfigs = {
            name: FilterConfig(name, attributes)
            for name, attributes in config["capture_criteria"].items()
        }

        capture_criteria = get_filters(capture_criteria_filter_configs)
        object.__setattr__(self, "capture_criteria", capture_criteria)


    def validate(self):
        # TODO: Validation
        print("TODO: Validate")


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


def get_filter_set(
    filter_command_line_args: Optional[Dict[str, Any]] = None,
    filter_config_from_file: Optional[Dict[str, Any]] = None,
    log: Logger = None,
) -> FilterSet:

    filter_config_from_file = (
        filter_config_from_file if filter_config_from_file is not None else {}
    )
    filter_command_line_args = (
        filter_command_line_args if filter_command_line_args is not None else {}
    )
    filter_command_line_args = stripped_by_keys(
        filter_command_line_args, ARG.FILTER.ALL
    )  # Only keep filter-related command line args
    # Command line args take precidence over configuration files in the event of a conflict.
    combined = {**filter_config_from_file, **filter_command_line_args}
    log.debug(f"\nFilters: {combined!s}")

    try:
        # Get the filter set name, then remove it from the dictionary so we know everything else in `combined `refers to actual filters.
        filter_set_name = combined.pop(ARG.FILTER.FILTER_SET_NAME)
    except KeyError as e:
        error_msg = f"Uh oh, we couldn't find the argument {ARG.FILTER.FILTER_SET_NAME} in either the config file or the command line. Please be sure to specify it in the config file, or on the command line."
        log.error(error_msg)
        raise e

    my_filter_configs = {
        filter_name: FilterConfig(filter_name, filter_attributes)
        for filter_name, filter_attributes in combined.items()
    }

    log.debug(f"\n{my_filter_configs!s}")

    filter_set = FilterSet.from_filter_configs(filter_set_name, my_filter_configs)
    return filter_set


def readconfig(path: PathLikeOrString, command_line_args: Optional[Dict[str, Any]] = None, log: Logger = getLogger()) -> PoretitionerConfig:
    """Read configuration from the path.

    Exceptions
    ----------
    FileNotFoundError: If path could not be resolved
    Parameters
    ----------
    path : Pathlike
        Path to the Poretitioner configuration file.
    """
    config_path = str(
        get_absolute_path(path)
    ).strip()  # Strip any trailing/leading whitespace.

    read_config = toml.load(config_path)
    # config = ConfigParser()

    gen_config = read_config[CONFIG.GENERAL]
    seg_config = read_config[CONFIG.SEGMENTATION]
    filter_config = read_config[CONFIG.FILTER]

    # config.read(config_path)
    # config = config
    log.debug(f"gen_config: {gen_config!s}")
    log.debug(f"seg_config: {seg_config!s}")
    log.debug(f"command_line_args: {command_line_args!s}")
    log.debug(f"filter_config: {filter_config!s}")

    filter_set = get_filter_set(
        filter_command_line_args=command_line_args,
        filter_config_from_file=filter_config,
        log=log,
    )

    segmentation_configuration = SegmentConfiguration(
        config=seg_config, command_line_args=command_line_args, log=log
    )
    general_configuration = GeneralConfiguration(
        config=gen_config, command_line_args=command_line_args, log=log
    )

    configs: PoretitionerConfig = {
        CONFIG.GENERAL: general_configuration,
        CONFIG.SEGMENTATION: segmentation_configuration,
        CONFIG.FILTER: filter_set,
        CONFIG.CLASSIFICATION: {}
    }

    return configs

    # TODO: Return configuration https://github.com/uwmisl/poretitioner/issues/73


def read_segmentation(config, command_line_args=None):
    seg_config = readconfig(config, command_line_args=command_line_args)
    print(seg_config)
