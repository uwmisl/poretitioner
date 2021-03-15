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
from typing import Any, Dict, List, Optional, Union

from json import JSONEncoder

import numpy as np

from ..getargs import ARG
from ..logger import getLogger, Logger

from .core import stripped_by_keys, PathLikeOrString

@dataclass(frozen=True)
class CONFIG:
    GENERAL = "general"
    SEGMENTATION = "segmentation"
    FILTER = "filters"


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

    def initialize_fields(self, command_line_args: Dict = None, config: Dict = None, log: Logger = None):
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
                log.warning(f"'{field}' is not a valid field for configuration {self.__class__.__name__!s}. Ignoring...")


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


class FilterJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            return vars(obj)
        except TypeError:
            pass
        # Best practice to let base class default raise the type error:
        # https://docs.python.org/3/library/json.html#json.JSONEncoder.default
        return super().default(obj)

@dataclass(frozen=True)
class FilterConfig:
    name: str
    attributes: Dict[str, Any]
    filepath: Optional[str] = None


@dataclass(frozen=True)
class FilterConfiguration:
    """
    Mapping of name to filter configuration.
    """
    filters: Dict[str, FilterConfig]


    def __init__(self, command_line_args: Dict = None, filter_config: Dict = None, log: Logger = getLogger()) -> None:
        # Rule of 3, this needs to be a helper of some kind
        filter_command_line_args = stripped_by_keys(command_line_args, ARG.FILTER.ALL) # Only keep filter-related command line args

        filter_config = filter_config if filter_config is not None else {}

        object.__setattr__(self, "filters", {})
        # Command line args take precidence over configuration files in the event of a conflict.
        combined = {**filter_config, **filter_command_line_args}
        log.debug(f"\nFilters: {combined!s}")

        for filter_name, filter_attributes in combined.items():
            # TODO: Filter Plugin should allow filepaths. https://github.com/uwmisl/poretitioner/issues/91
            filepath = None
            filter_config = FilterConfig(filter_name, filter_attributes, filepath)
            log.debug(f"\nFilters[{filter_name}] = {filter_config!r}")
            self.filters[filter_name] = filter_config

    def validate(self):
        raise NotImplementedError("Implement validation for filters!")

    def json_encoder(self) -> JSONEncoder:
        encoder = FilterJSONEncoder()
        return encoder

    @classmethod
    def from_json(cls, json_dict: Dict):
        filters = {
            filter_config.get("name"): FilterConfig(**filter_config) for filter_config in json_dict["filters"] 
        }
        return cls.__new__(filters)


@dataclass(frozen=True)
class GeneralConfiguration(BaseConfiguration):
    n_workers: int
    capture_directory: str

    def validate(self):
        # assert self.n_workers > 0
        # assert self.captures_per_f5 > 0
        pass

    def __init__(self, command_line_args: Dict = None, config: Dict = None, log: Logger = getLogger()) -> None:
        """[summary]

        Parameters
        ----------
        command_line_args : Dict, optional
            [description], by default None
        config : Dict, optional
            [description], by default None
        """
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
    good_channels: List[int]
    open_channel_prior_mean: int
    open_channel_prior_stdv: int
    terminal_capture_only: bool

    def __init__(self, command_line_args: Dict = None, config: Dict = None, log: Logger = getLogger()) -> None:
        """[summary]

        Parameters
        ----------
        command_line_args : Dict, optional
            Command line arguments for filters, by default None. Any keys outside of 
            ARG.FILTER are ignored. 
        config : Dict, optional
            Segmentation configuration, by default None
        """
        command_line_args = stripped_by_keys(command_line_args, self.valid_field_names) # Only keep filter-related command line args
        self.initialize_fields(command_line_args=command_line_args, config=config)

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


def readconfig(path, command_line_args=None, log: Logger = getLogger()):
    """Read configuration from the path.

    Exceptions
    ----------
    FileNotFoundError: If path could not be resolved
    Parameters
    ----------
    path : Pathlike
        Path to the Poretitioner configuration file.
    """
    config_path = str(get_absolute_path(path)).strip() # Strip any trailing/leading whitespace.

    read_config = toml.load(config_path)
    #config = ConfigParser()

    gen_config = read_config[CONFIG.GENERAL]
    seg_config = read_config[CONFIG.SEGMENTATION]
    filter_config = read_config[CONFIG.FILTER]
 
    #config.read(config_path)
    #config = config
    log.debug(f"\n\ngen_config: {gen_config!s}\n\n")
    log.debug(f"\n\nseg_config: {seg_config!s}\n\n")
    log.debug(f"\n\ncommand_line_args: {command_line_args!s}\n\n")
    log.debug(f"\n\nfilter_config: {filter_config!s}\n\n")

    filter_commandline_args = {}
    filter_configuration = FilterConfiguration(filter_config=filter_config, command_line_args=command_line_args, log=log)

    segmentation_configuration = SegmentConfiguration(config=seg_config, command_line_args=command_line_args, log=log)
    general_configuration = GeneralConfiguration(config=gen_config, command_line_args=command_line_args, log=log)


    configs = {
        CONFIG.GENERAL: general_configuration,
        CONFIG.SEGMENTATION: segmentation_configuration,
        CONFIG.FILTER: filter_configuration,
    }

    return configs

    # TODO: Return configuration https://github.com/uwmisl/poretitioner/issues/73


def read_segmentation(config, command_line_args=None):
    seg_config = readconfig(config, command_line_args=command_line_args)
    print(seg_config)
