"""
=========
filtering.py
=========

This module provides more granular filtering for captures.
You can customize your own filters too.

"""
import re
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import PosixPath
from typing import Any, Dict, List, NewType, Optional, Union

import h5py
import numpy as np
from json import JSONEncoder

from ..logger import Logger, getLogger
from ..signals import Capture
from .core import NumpyArrayLike, PathLikeOrString
from .core import ReadId
from .core import stripped_by_keys

CaptureOrTimeSeries = Union[Capture, NumpyArrayLike]

# Unique identifier for a collection of filters (e.g. "ProfJeffsAwesomeFilters")
FilterSetId = NewType("FilterSetId", str)

# Unique identifier for an individual filter (e.g. "min_frac")
FilterName = NewType("FilterName", str)


__all__ = [
    "does_pass_filters",
    "get_filters",
    "FilterName",
    "FilterSetId",
    "FilterConfig",
    "Filter",
    "Filters",
    "DEFAULT_FILTER_PLUGINS"
    "FilterSet",
    "FilterConfigs",
    "FilterPlugin",
    "PATH",
]

@dataclass(frozen=True)
class PATH:
    ROOT = f"/Filter/"

    @classmethod
    def filter_path_for_filter_set(cls, filter_set: FilterSetId) -> str:
        filter_path = str(PosixPath(PATH.ROOT, filter_set))
        return filter_path

    @classmethod
    def filter_pass_path_for_read_id(cls, read_id: ReadId) -> str:
        pass_path = str(PosixPath(PATH.ROOT, "pass", read_id))
        return pass_path

    def get_filter_pass_path(read_id):
        path = str(PosixPath(PATH.ROOT, "pass"))
        return path

@dataclass(frozen=True)
class FilterConfig:
    """A blueprint for how to construct a FilterPlugin.

    Note on terminology:
    
        - FilterConfig: A high-level description of a filter.
    
        - FilterPlugin: An actual, callable, implementation of a FilterConfig.


    For custom plugins, make sure "filepath" is an attribute that points to the file to laod
    """
    name: str
    attributes: Dict[str, Any]
    

# Mapping of a FilterName to filter configurations.
FilterConfigs = NewType("FilterConfigs", Dict[FilterName, FilterConfig])

# TODO: Filter Plugin should check that name is unique. https://github.com/uwmisl/poretitioner/issues/91
class FilterPlugin(metaclass=ABCMeta):
    """
    Abstract class for Filter plugins. To write your own filter, subclass this abstract
    class and implement the `apply` method and `name` property.
    """

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Unique name for this filter.
        Make sure it doesn't conflict with any existing names.

        Returns
        -------
        str
            The unique name for this filter (e.g. "fourier_transform").

        Raises
        ------
        NotImplementedError
            Raised if this filter is called without this name method being implemented.
        """
        raise NotImplementedError(
            "'name' class method not implemented for filter. This class method should return a unique name for this filter."
        )

    @abstractmethod
    def apply(self, capture: CaptureOrTimeSeries) -> bool:
        """Returns True if a capture passes a given filter criteria.
        For instance, a range filter would check that a capture's summary statistsics lie within a given range.

        Parameters
        ----------
        capture : np.typing.ArrayLike
            Time series capture to filter.

        Returns
        -------
        bool
            Whether this capture passes the filter.

        Raises
        ------
        NotImplementedError
            Raised when the filter method isn't implemented by the consuming Filter class
        """
        raise NotImplementedError(
            "'apply' method not implemented for filter. This method should return True if and only if applied to a capture that meets the filter criterion. For instance, "
        )

    def __call__(self, capture: CaptureOrTimeSeries) -> bool:
        """Apply the filter.

        Defining `__call__` lets us do nice things like:

        class MyCustomFilter(FilterPlugin):
            def apply(capture):
                # ...
                pass

        # Later in code where filtering is done....

        valid_captures = []
        filters = [ MyCustomFilter(), AnotherCustomFilter(), ... ]

        valid_captures = [capture for capture in captures if all([filt(capture) for filt in filters])]
        for capture in captures: # You'd want to parallelize this in a real life example...

            for filt in filters:
                filtered_captures = filt(capture).

        Parameters
        ----------
        capture : CaptureOrTimeSeries
            Capture to filter.

        Returns
        -------
        bool
            Whether this capture passes the filter.
        """
        result = self.apply(capture)
        return result


class RangeFilter(FilterPlugin):

    def __init__(self, minimum: Optional[float] = None, maximum: Optional[float] = None):
        """A filter that filters based on whether a signal falls between a maximum and a minimum.

        Parameters
        ----------
        minimum : float, optional
            The smallest value this signal should be allowed to take (inclusive), by default RangeFilter.DEFAULT_MINIMUM
        maximum : float, optional
            The largest value this signal should be allowed to take (inclusive), by default RangeFilter.DEFAULT_MAXIMUM
        """
        self.minimum = minimum
        self.maximum = maximum

    def extract(self, capture: CaptureOrTimeSeries) -> Any:
        """Extracts a summary statistic from the capture (e.g. mean, length, standard deviation).

        Identity operation by default (just returns the capture).

        You can use this function to transform the data in a useful way before processing it (e.g.
        getting the mean value of a capture before filtering based on that mean.)

        Note: If we picture the filtering workflow as an ETL (Extract-Transform-Load) pipeline, this would be the "transform"
        (take data, modify it for a later purpose), but I feel that "transform" is perhaps a misleading function name in this context.

        Parameters
        ----------
        capture : CaptureOrTimeSeries
            Capture from which to extract data.
        """
        return capture

    def is_in_range(self, value: float) -> bool:
        minimum = self.minimum if self.minimum is not None else -np.inf
        maximum = self.maximum if self.maximum is not None else np.inf

        return minimum <= value <= maximum

    def apply(self, signal):
        value = self.extract(signal)
        return self.is_in_range(value)


class StandardDeviationFilter(RangeFilter):
    """ Filters for captures with standard deviations in some range.
    """

    @classmethod
    def name(cls) -> str:
        return "stdv"

    def extract(self, capture: CaptureOrTimeSeries):
        signal = capture
        try:
            signal = capture.fractionalized
        except AttributeError:
            pass
        return np.std(signal)


class MeanFilter(RangeFilter):
    """Filters for captures with an arithmetic mean within a range.
    """

    @classmethod
    def name(cls) -> str:
        return "mean"

    def extract(self, capture: CaptureOrTimeSeries):
        signal = capture
        try:
            signal = capture.fractionalized
        except AttributeError:
            pass
        return np.mean(signal)


class MedianFilter(RangeFilter):
    @classmethod
    def name(cls) -> str:
        return "median"

    def extract(self, capture: CaptureOrTimeSeries):
        signal = capture
        try:
            signal = capture.fractionalized
        except AttributeError:
            pass
        return np.median(signal)


class MinimumFilter(RangeFilter):
    @classmethod
    def name(cls) -> str:
        return "min"

    def extract(self, capture: CaptureOrTimeSeries):
        signal = capture
        try:
            signal = capture.fractionalized
        except AttributeError:
            pass
        return np.min(signal)


class MaximumFilter(RangeFilter):
    @classmethod
    def name(cls) -> str:
        return "max"

    def extract(self, capture: CaptureOrTimeSeries):
        signal = capture
        try:
            signal = capture.fractionalized
        except AttributeError:
            pass
        return np.max(signal)


class LengthFilter(RangeFilter):
    """Filters captures based on their length.
    """

    @classmethod
    def name(cls) -> str:
        return "length"

    def extract(self, capture: CaptureOrTimeSeries):
        signal = capture
        try:
            signal = capture.fractionalized
        except AttributeError:
            pass
        return len(signal)


"""
How to Create Your Own Custom Filter:

Need more advanced filtering than what we provide out of the box? No problem.
Create your own custom filter by inheriting from the FilterPlugin class.

For this example, let's do something complex. Say you only want to examine captures
that have more than 5 samples with a hyperbolic tangent greater than some threshold.

That means our custom filter's `apply` function should return True if and only if
the signal has more than 5 samples greater than the threshold, after taking the hyperbolic tangent in `extract`.
"""


class MyCustomFilter(FilterPlugin):
    threshold: float = 0.5  # Totally arbitrary.

    def extract(self, capture):
        # Do the transformations here, or pre-process it before the filter.

        # Gets the hyperbolic tangent of the signal.
        extracted = np.tanh(capture.signal)
        return extracted

    def apply(self, signal):
        # Only return true if more than 5 samples have a square root greater than 2.0 (arbitrary)
        extracted = self.extract(signal)

        # If we want to filter out signals with fewer than 5 matching samples, then we
        # should retrun True when there are 5 or more matching samples.
        n_meeting_threshold = len(
            extracted[extracted > self.threshold]
        )  # Number of samples greater than the threshold
        meets_criteria = (
            n_meeting_threshold >= 5
        )  # Are there at least 5 samples meeting this threshold?
        return meets_criteria


def check_capture_ejection_by_read(f5, read_id):
    """Checks whether the current capture was in the pore until the voltage
    was reversed.

    Parameters
    ----------
    f5 : TODO
    read_id : TODO

    Returns
    -------
    boolean
        True if the end of the capture coincides with the end of a voltage window.
    """
    try:
        ejected = f5.get(f"/read_{read_id}/Signal").attrs["ejected"]
    except AttributeError:
        raise ValueError(f"path /read_{read_id} does not exist in the fast5 file.")
    return ejected


def check_capture_ejection(end_capture, voltage_ends, tol_obs=20):
    """Checks whether the current capture was in the pore until the voltage
    was reversed.

    Essentially checks whether a value (end_capture) is close enough (within
    a margin of tol_obs) to any value in voltage_ends.

    Parameters
    ----------
    end_capture : numeric
        The end time of the capture.
    voltage_ends : list of numeric
        List of times when the standard voltage ends.
    tol_obs : int, optional
        Tolerance for defining when the end of the capture = voltage end, by default 20

    Returns
    -------
    boolean
        True if the end of the capture coincides with the end of a voltage window.
    """
    for voltage_end in voltage_ends:
        if np.abs(end_capture - voltage_end) < tol_obs:
            return True
    return False



# def apply_filters_to_read(config, f5, read_id, filter_name):
#     passed_filters = True

#     # Check whether the capture was ejected
#     if "ejected" in config["filters"][filter_name]:
#         only_use_ejected_captures = config["filters"][filter_name]["ejected"]  # TODO
#         if only_use_ejected_captures:
#             capture_ejected = check_capture_ejection_by_read(f5, read_id)
#             if not capture_ejected:
#                 passed_filters = False
#                 return passed_filters
#     else:
#         only_use_ejected_captures = False  # could skip this, leaving to help read logic

#     # Apply all the filters
#     get_raw_signal()
#     signal = raw_signal_utils.get_fractional_blockage_for_read(f5, read_id)
#     # print(config["filters"][filter_name])
#     # print(f"min = {np.min(signal)}")
#     passed_filters = apply_feature_filters(signal, config["filters"][filter_name])
#     return passed_filters


def filter_and_store_result(config, fast5_files, filter_name, overwrite=False):
    # Apply a new set of filters
    # Write filter results to fast5 file (using format)
    # if only_use_ejected_captures = config["???"]["???"], then check_capture_ejection
    # if there's a min length specified in the config, use that
    # if feature filters are specified, apply those
    # save all filter parameters in the filter_name path
    filter_path = f"/Filter/{filter_name}"

    # TODO: parallelize this (embarassingly parallel structure)
    for fast5_file in fast5_files:
        with h5py.File(fast5_file, "a") as f5:
            if overwrite is False and filter_path in f5:
                continue
            passed_read_ids = []
            for read_h5group_name in f5.get("/"):
                if "read" not in read_h5group_name:
                    continue
                read_id = re.findall(r"read_(.*)", read_h5group_name)[0]

                passed_filters = apply_filters_to_read(config, f5, read_id, filter_name)
                if passed_filters:
                    passed_read_ids.append(read_id)
            write_filter_results(f5, config, passed_read_ids, filter_name)


#  def write_filter_results(f5, passing_read_ids: List[str], log):
#         # For all read_ids that passed the filter (AKA reads that were passed in),
#         # create a hard link in the filter_path to the actual read's location in
#         # the fast5 file.
#         for read_id in passing_read_ids:
#             read_path = format_read_id(read_id)
#             read_grp = f5.get(read_path)
#             self.log.debug(read_grp)
#             filter_read_path = FILTER_PATH.filter_pass_path_for_read_id(read_id)
#             # Create a hard link from the filter read path to the actual read path
#             f5[filter_read_path] = read_grp


def filter_like_existing(config, example_fast5, example_filter_path, fast5_files, new_filter_path):
    # Filters a set of fast5 files exactly the same as an existing filter
    # TODO : #68 : implement
    raise NotImplementedError()


__DEFAULT_FILTER_PLUGINS = [
    MeanFilter,
    StandardDeviationFilter,
    MedianFilter,
    MinimumFilter,
    MaximumFilter,
    LengthFilter,
]

DEFAULT_FILTER_PLUGINS = { filter_plugin_class.name() : filter_plugin_class for filter_plugin_class in __DEFAULT_FILTER_PLUGINS }

@dataclass
class Filter:
    """A named filter that can be applied to some data.

    You can use this filter by just calling it on some data.

    my_signal = [1,2,3,4]

    filter = Filter(...)

    passed_filter: bool = filter(my_signal)

    Parameters
    ----------
    config : FilterConfig
        A description of this filter's configuration (e.g. where it was loaded from).
    plugin : FilterPlugin
        The actual implementation of this filter.
        We have this class defined with 
    """

    config: FilterConfig
    plugin: FilterPlugin
        
    def __call__(self, *args, **kwargs):
        return self.plugin(*args, **kwargs)

    def apply(self, *args, **kwargs):
        self.plugin.apply(*args, **kwargs)
    
    @property
    def name(self) -> FilterName:
        return self.plugin.name()


@dataclass
class Filters:
    """A collection of callable filters and their names.

    Use this like a dictionary of 

    This is probably what you want to use and pass around.

    """
    _filters: Dict[FilterName, Filter]
    
    def __init__(self, filters: Dict[FilterName, Filter]):
        self._filters = filters

    def __getitem__(self, filter_name: str):
        return self._filters[filter_name]
    
    def __setitem__(self, filter_name: str):
        return self._filters[filter_name]

    def values(self):
        return self._filters.values()

    def items(self):
        return self._filters.items()

    # def filters(self):
    #     return self._filters.values()

    # def __dask_tokenize__(self):
    #     # For tokenize to work we want to return a value that fully
    #     # represents this object. In this case it's the config dictionary.
    #     return self._filters
    

    def __call__(self, capture: CaptureOrTimeSeries):
        """
        Check whether an array of values (e.g. a single nanopore capture)
        passes a set of filters.

        Parameters
        ----------
        capture : CaptureOrTimeSeries | NumpyArrayLike
            Capture containing time series of nanopore current values for a single capture, or the signal itself.

        Returns
        -------
        boolean
            True if capture passes all filters; False otherwise.
        """
        return does_pass_filters(capture, self._filters.values())


def get_filters(filter_configs: Optional[FilterConfigs] = None) -> Filters:
    """Creates Filters from a list of filter configurations.

    Parameters
    ----------
    filter_configs :  Optional[FilterConfigs]
        A mapping of filter names to their configurations, None by default (i.e. no filtering).

    Returns
    -------
    Filters
        A set of callable/applyable filters.
    """
    filter_configs = filter_configs if filter_configs is not None else {}
    my_filters = Filters({ name : filter_from_config(filter_config) for name, filter_config in filter_configs.items() })
    return my_filters

@dataclass(frozen=True)
class FilterSet:
    """
    A collection of filters with a name for easy
    identification. 
    Mapping of filter_set_name to its filters.
    """
    name: FilterSetId
    filters: Filters

    def validate(self):
        raise NotImplementedError("Implement validation for filters!")

    def json_encoder(self) -> JSONEncoder:
        encoder = FilterJSONEncoder()
        return encoder

    @classmethod
    def from_json(cls, filter_set_name: FilterSetId, filters_dict: Dict):
        filters = {
            filter_config.get("name"): FilterConfig(**filter_config) for filter_config in json_dict["filters"] 
        }
        return cls.__new__(filter_set_name, filters)
    
    @classmethod
    def from_filter_configs(cls, name: FilterSetId, filter_configs: FilterConfigs = None):
        filters: Filters = get_filters(filter_configs)
        filter_set = cls.__new__(cls)
        filter_set.__init__(name, filter_configs)
        return filter_set


def filter_from_config(config: FilterConfig, log: Logger = getLogger()) -> Filter:
    """Creates a Filter from a config spefication. If no "filename" is present in the FilterConfig, it's 
    assumed to be one of the default filtesr 

    Parameters
    ----------
    config : FilterConfig
        Filter configuration to build the plugin.
    log : Logger, optional
        Logger to use for information/warnings/debug, by default getLogger()

    Returns
    -------
    Filter
        A filter that can be applied to some data.

    Raises
    ------
    AttributeError
        A filter plugin could not be built from the configuration description. If this error is raised, be sure to check
        1) A plugin class with the name in the configuration is defined at the filepath described in the configuration
        2) The plugin class inherits from the `FilterPlugin` abstract base class.
    """
    name = config.name
    
    attributes: Dict[str, Any] = config.attributes
    filepath = attributes.get("filepath", None)

    # TODO: For non-default FilterPlugins, load/unpickle the class from the filepath. https://github.com/uwmisl/poretitioner/issues/91
    plugin = None

    if name in DEFAULT_FILTER_PLUGINS:
        plugin = DEFAULT_FILTER_PLUGINS[name]()
    else:
        # TODO: For non-default FilterPlugins, load the class from the filepath. https://github.com/uwmisl/poretitioner/issues/91
        plugin = plugin_from_file(name, filepath)
        pass

    # Make sure any plugin attributes defined in the config are moved over to the plugin instance.
    try:
        # Here, we take care of setting whatever attributes the plugin config defines on the new plugin instance.
        for key, value in attributes.items():
            object.__setattr__(plugin, key, value)
    except AttributeError as e:
        log.warning(
            """
        Uh oh, couldn't find plugin '{name}'. Are you sure:

        1) A plugin class with the name '{name}' is defined in the file {filepath}?

        2) That plugin class inherits from `FilterPlugin`?
        """
        )
        raise e

    my_filter = Filter(config, plugin)

    return my_filter


def plugin_from_file(name: str, filepath: PathLikeOrString):
    """[summary]

    Parameters
    ----------
    name : str
        [description]
    filepath : PathLikeOrString
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """
    # TODO: For non-default FilterPlugins, load/unpickle the class from the filepath. https://github.com/uwmisl/poretitioner/issues/91
    raise NotImplementedError("Plugin from file has not been implemented! This method should take in a filepath and filter name, and return a runnable FilterPlugin!")


class FilterJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Filters):
            return obj.items()
        try:
            return vars(obj)
        except TypeError:
            pass
        # Best practice to let base class default raise the type error:
        # https://docs.python.org/3/library/json.html#json.JSONEncoder.default
        return super().default(obj)


def does_pass_filters(capture: CaptureOrTimeSeries, filters: Filters) -> bool:
    """
    Check whether an array of values (e.g. a single nanopore capture)
    passes a set of filters. Filters can be based on summary statistics
    (e.g., mean) and/or a range of allowed values.

    Parameters
    ----------
    capture : CaptureOrTimeSeries | NumpyArrayLike
        Capture containing time series of nanopore current values for a single capture, or the signal itself.
    filters : Filters
        The set of filters to apply. Write your own filter by subclassing FilterPlugin.

    Returns
    -------
    boolean
        True if capture passes all filters; False otherwise.
    """
    # TODO: Parallelize? https://github.com/uwmisl/poretitioner/issues/67
    all_passed = True 
    for filter_out in filters.values(): 
        if not filter_out(capture):
            return False 
    return True    
