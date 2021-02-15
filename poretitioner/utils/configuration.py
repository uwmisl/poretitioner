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
from configparser import ConfigParser
from dataclasses import dataclass
from distutils.utils import strtobool
from pathlib import Path
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class SegmentConfiguration:
    bulkfast5: str
    segmentedfast5: str
    voltage_threshold: float
    signal_threshold_frac: float
    translocation_delay: float
    alt_open_channel_pA: float
    terminal_capture_only: bool
    delay: int
    end_tol: float
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = True
    filters = {"length": (100, None)}
    delay = 0
    end_tol = 0

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


@dataclass(frozen=True)
class FilterConfiguration:
    # Filters to apply to the data.
    # Valid filters take the form:
    # "name":
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
    median_min: float = 0.15
    # Maximum signal median to pass. Defaults to +∞ if not present.
    median_max: float = 1

    use_minimum: bool = True
    # Minimum signal absolute value to pass. Defaults to -∞ if not present.
    minimum: float = -math.inf

    use_max: bool = True
    # Maximum signal absolute value to pass. Defaults to +∞ if not present.
    maximum: float = math.inf

    use_length: False
    length: float = math.inf


@dataclass(frozen=True)
class QuantifierConfiguration:
    pass


@dataclass(frozen=True)
class ClassifierConfiguration:
    pass


def number(num_str):
    if num_str == "∞":
        return math.inf
    elif num_str == "-∞":
        return -math.inf
    return float(num_str)


def str_to_bool(string):
    if string.upper() == "TRUE":
        return True
    elif string.upper() == "FALSE":
        return False
    raise ValueError(
        f"'{string}' is not parsible into 'True' or 'False'. Double-check that it's spelled correctly."
    )


def get_absolute_path(path):
    absolute_path = Path(path).expanduser().resolve()


def readconfig(path):
    """[summary]

    Exceptions
    ----------
    FileNotFoundError: If path could not be resolved
    Parameters
    ----------
    path : [type]
        [description]


    """
    config = ConfigParser()
    config.read(Path(path).resolve())
    config = config


def read_segmentation(config):
    readconfig(config)


if __name__ == "__main__":
    path = Path("/Users/dna/Developer/poretitioner/config.ini")
    read_segmentation(path)
