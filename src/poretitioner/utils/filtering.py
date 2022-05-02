"""
=========
filtering.py
=========

This module provides more granular filtering for captures.
You can customize your own filters too.

"""
from __future__ import annotations

import re
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from json import JSONEncoder
from pathlib import PosixPath
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    NewType,
    Optional,
    Protocol,
    Type,
    TypedDict,
    Union,
)

import h5py
import numpy as np
from h5py import File as Fast5File

from ..hdf5 import (
    HasFast5,
    HDF5_Group,
    HDF5_GroupSerialableDataclass,
    HDF5_GroupSerializable,
    HDF5_GroupSerializing,
    IsAttr,
)
from ..logger import Logger, getLogger
from ..signals import Capture
from .core import NumpyArrayLike, PathLikeOrString, ReadId, stripped_by_keys
from .plugin import Plugin

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
    "DEFAULT_FILTER_PLUGINS",
    "FilterSet",
    "FilterConfigs",
    "FilterPlugin",
    "PATH",
]


@dataclass(frozen=True)
class FILTER_PATH:
    ROOT = f"/Filter/"

    @classmethod
    def filter_set_path(cls, filter_set_id: FilterSetId) -> str:
        filter_path = str(PosixPath(FILTER_PATH.ROOT, filter_set_id))
        return filter_path

    @classmethod
    def filter_set_pass_path(cls, filter_set_id: FilterSetId) -> str:
        pass_path = str(PosixPath(FILTER_PATH.filter_set_path(filter_set_id), "pass"))
        return pass_path

    @classmethod
    def filter_set_pass_path_for_read_id(cls, filter_set_id: FilterSetId, read_id: ReadId) -> str:
        pass_path = str(PosixPath(FILTER_PATH.filter_set_pass_path(filter_set_id), read_id))
        return pass_path


class FilterConfig(TypedDict):
    """A blueprint for how to construct a FilterPlugin.

    Contains a name, and any number of other attributes

    Note on terminology:

        - FilterConfig: A high-level description of a filter.

        - FilterPlugin: An actual, callable, implementation of a FilterConfig.


    For custom plugins, make sure "filepath" is an attribute that points to the file to laod
    """


# Mapping of a FilterName to filter configurations.
FilterConfigs = NewType("FilterConfigs", Dict[FilterName, FilterConfig])

# TODO: Filter Plugin should check that name is unique. https://github.com/uwmisl/poretitioner/issues/91
class FilterPlugin(Plugin):
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


RANGE_FILTER_DEFAULT_MINIMUM: float = -np.inf
RANGE_FILTER_DEFAULT_MAXIMUM: float = np.inf


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
        self.minimum = minimum if minimum is not None else RANGE_FILTER_DEFAULT_MINIMUM
        self.maximum = maximum if maximum is not None else RANGE_FILTER_DEFAULT_MAXIMUM

    def extract(self, capture: CaptureOrTimeSeries) -> NumpyArrayLike:
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
        try:
            signal = capture.fractionalized()
        except AttributeError:
            signal = capture
        else:
            signal = capture
        return signal
        # signal = getattr(capture, Capture.fractionalized.__name__, capture)

    def is_in_range(self, value: Union[NumpyArrayLike, float]) -> bool:
        try:
            # If the value is just a float, we can use this handy syntax:
            return self.minimum <= value <= self.maximum
        except ValueError:
            # But we're not allowed to use that syntax on numpy arrays.
            return all(np.logical_and(self.minimum <= value, value <= self.maximum))

    def apply(self, signal):
        value = self.extract(signal)
        return self.is_in_range(value)


class StandardDeviationFilter(RangeFilter):
    """Filters for captures with standard deviations in some range."""

    @classmethod
    def name(cls) -> str:
        return "stdv"

    def extract(self, capture: CaptureOrTimeSeries):
        signal = super().extract(capture)
        return np.std(signal)


class MeanFilter(RangeFilter):
    """Filters for captures with an arithmetic mean within a range."""

    @classmethod
    def name(cls) -> str:
        return "mean"

    def extract(self, capture: CaptureOrTimeSeries):
        signal = super().extract(capture)
        return np.mean(signal)


class MedianFilter(RangeFilter):
    """Filters for captures with a median within a range."""

    @classmethod
    def name(cls) -> str:
        return "median"

    def extract(self, capture: CaptureOrTimeSeries):
        signal = super().extract(capture)
        return np.median(signal)


class MinimumFilter(RangeFilter):
    """Filters for captures with a minimum within a range."""

    @classmethod
    def name(cls) -> str:
        return "min"

    def extract(self, capture: CaptureOrTimeSeries):
        signal = super().extract(capture)
        return np.min(signal)


class MaximumFilter(RangeFilter):
    """Filters for captures with a maximum within a range."""

    @classmethod
    def name(cls) -> str:
        return "max"

    def extract(self, capture: CaptureOrTimeSeries):
        signal = super().extract(capture)
        return np.max(signal)


class LengthFilter(RangeFilter):
    """Filters captures based on their length."""

    @classmethod
    def name(cls) -> str:
        return "length"

    def extract(self, capture: CaptureOrTimeSeries):
        signal = super().extract(capture)
        return len(signal)


class EjectedFilter(FilterPlugin):
    """Filters captures based on whether they were ejected from the pore."""

    @classmethod
    def name(cls) -> str:
        return "ejected"

    def extract(self, capture: Capture):
        return capture.ejected


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

    def name(self):
        return "foo"

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


def apply_feature_filters(capture: CaptureOrTimeSeries, filters: List[FilterPlugin]) -> bool:
    """
    Check whether an array of current values (i.e. a single nanopore capture)
    passes a set of filters. Filters can be based on summary statistics
    (e.g., mean) and/or a range of allowed values.

    Notes on filter behavior: If the filters list is empty, there are no filters
    and the capture passes.

    Parameters
    ----------
    capture : CaptureOrTimeSeries | NumpyArrayLike
        Capture containing time series of nanopore current values for a single capture, or the signal itself.
    filters : List[FilterPlugin]
        List of FilterPlugin instances. Write your own filter by subclassing FilterPlugin.

    Returns
    -------
    boolean
        True if capture passes all filters; False otherwise.
    """
    if filters is None:
        filters = []

    # TODO: Parallelize? https://github.com/uwmisl/poretitioner/issues/67
    filtered = [filter_out(capture) for filter_out in filters]
    print(filtered)

    # Did this signal pass all filters?
    all_passed = all(filtered)
    return all_passed


def check_capture_ejection_by_read(f5, read_id):
    """Checks whether the current capture was in the pore until the voltage
    was reversed.

    Parameters
    ----------
    f5 : h5py.File object (open for reading or more)
        Capture fast5 file
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


def filter_like_existing(config, example_fast5, example_filter_path, fast5_files, new_filter_path):
    # Filters a set of fast5 files exactly the same as an existing filter
    # TODO : #68 : implement
    raise NotImplementedError()


def get_filter_pass_path(filter_set_id, read_id):
    return FILTER_PATH.filter_set_pass_path(filter_set_id)


__DEFAULT_FILTER_PLUGINS = [
    MeanFilter,
    StandardDeviationFilter,
    MedianFilter,
    MinimumFilter,
    MaximumFilter,
    LengthFilter,
]

DEFAULT_FILTER_PLUGINS = {
    filter_plugin_class.name(): filter_plugin_class
    for filter_plugin_class in __DEFAULT_FILTER_PLUGINS
}


class Filtering(Protocol):
    """Classes that adhere to the Filtering protocol
    provide an 'apply' method to an input that returns True
    if and only if the input passes its filter.

    These are also callable, so calling a filter on an input
    is functionally equivalent to calling its apply method.
    """

    def __call__(self, *args, **kwargs) -> bool:
        raise NotImplementedError("Filtering protocol hasn't implemented __call__ yet!")

    def apply(self, *args, **kwargs) -> bool:
        raise NotImplementedError("Filtering protocol hasn't implemented Apply yet!")


@dataclass
class Filter(Filtering):
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

    def __call__(self, *args, **kwargs) -> bool:
        return self.plugin(*args, **kwargs)

    def apply(self, *args, **kwargs) -> bool:
        return self.plugin.apply(*args, **kwargs)

    @property
    def name(self) -> FilterName:
        return FilterName(self.plugin.__class__.name())

    def as_attr(self) -> Dict[str, Any]:
        name = self.name
        attrs = {**vars(self.config), **vars(self.plugin), name: name}
        return attrs

    def from_attr(self, attr) -> IsAttr:
        ...


import json


@dataclass
class HDF5_FilterSerialable(Filter, HDF5_GroupSerialableDataclass):
    def add_to_group(self, parent_group: HDF5_Group, log: Optional[Logger] = None) -> HDF5_Group:
        log = log if log is not None else getLogger()
        # Note: This line simply registers a group with the name 'name' in the parent group.
        this_group = HDF5_Group(parent_group.require_group(self.name))

        all_attrs = {**self.config, **vars(self.plugin)}
        this_group.create_attrs(all_attrs)

        # Implementers must now write their serialized instance to this group.
        return this_group

    @classmethod
    def from_group(
        cls, group: HDF5_Group, log: Optional[Logger] = None
    ) -> HDF5_GroupSerialableDataclass:
        # You see, the trouble is, in the above 'add_to_group' call, we lumped together
        # all the attributes of the FilterConfig and the FilterPlugin, not knowing
        # which attributes belonged to which class.
        #
        # Now, here in `from_group`, it's time to pay the piper and figure out which attribute
        # goes where to create a new Filter instance.
        #
        # This is likely achievable through the plugin architecture, since the plugin's
        # name is unique, we can try to find a plugin with a given name, then get its attributes from there.
        # Load
        log.warning("Filter.from_group not implemented...It's a whole thing (see comment)")

        # This is pure Hail Mary.
        return super().from_group(group, log)


# class Filters(HDF5_GroupSerialableDataclass):
#     filters:


Filters = Dict[FilterName, Filter]


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
    filter_configs = filter_configs if filter_configs is not None else FilterConfigs({})
    my_filters = {
        name: filter_from_config(name, filter_config)
        for name, filter_config in filter_configs.items()
    }

    return my_filters


def does_pass_filters(capture: CaptureOrTimeSeries, filters: Iterable[Filter]) -> bool:
    """
    Check whether an array of values (e.g. a single nanopore capture)
    passes a set of filters. Filters can be based on summary statistics
    (e.g., mean) and/or a range of allowed values.

    Parameters
    ----------
    capture : CaptureOrTimeSeries | NumpyArrayLike
        Capture containing time series of nanopore current values for a single capture, or the signal itself.
    filters : Iterable[Filter]
        The set of filters to apply. Write your own filter by subclassing FilterPlugin.

    Returns
    -------
    boolean
        True if capture passes all filters; False otherwise.
    """
    all_passed = True
    for some_filter in filters:
        if not some_filter(capture):
            return False
    return all_passed


@dataclass(frozen=True)
class FilterSetProtocol(Filtering, Protocol):
    filter_set_id: FilterSetId
    filters: Filters

    @classmethod
    def from_filter_configs(cls, name: FilterSetId, filter_configs: FilterConfigs = None):
        ...


@dataclass(frozen=True, init=False)
class FilterSet(FilterSetProtocol):
    """
    A collection of filters with a name for easy
    identification. Essentially a mapping of filter names to their implementations.
    """

    def validate(self):
        raise NotImplementedError("Implement validation for filters!")

    def __init__(self, filter_set_id: FilterSetId, filters: Filters) -> None:
        filterset = super().__init__(self)
        object.__setattr__(self, "filter_set_id", filter_set_id)
        object.__setattr__(self, "filters", filters)
        # self.name = name
        # self.filters = filters

    ############################
    #
    #   FilterSetProtocol
    #
    ############################
    @classmethod
    def from_filter_configs(cls, name: FilterSetId, filter_configs: FilterConfigs = None):
        filters: Filters = get_filters(filter_configs)
        filter_set = cls.__new__(cls, name, filters)
        filter_set.__init__(name, filters)
        return filter_set

    def apply(self, capture: CaptureOrTimeSeries) -> bool:
        return does_pass_filters(capture, self.filters.values())

    def __call__(self, capture: CaptureOrTimeSeries) -> bool:
        return self.apply(capture)


class HDF5_FilterSet(FilterSet, HDF5_GroupSerialableDataclass):
    def __init__(self, filter_set: FilterSet) -> None:
        self._filterset = filter_set

    ############################
    #
    #   HDF5_GroupSerializable
    #
    ############################

    def name(self):
        return self._filterset.filter_set_id

    def add_to_group(self, parent_group: HDF5_Group, log: Optional[Logger] = None) -> HDF5_Group:
        filter_set_group = parent_group.require_group(self.name())
        for name, filter_t in self._filterset.filters.items():
            hdf5_filter = HDF5_FilterSerialable(filter_t.config, filter_t.plugin)
            hdf5_filter.add_to_group(filter_set_group)

        return HDF5_Group(filter_set_group)

    # @classmethod
    # def from_group(
    #     cls, group: HDF5_Group, log: Optional[Logger] = None
    # ) -> HDF5_GroupSerializable:
    #     raise NotImplementedError(
    #         f"from_group not implemented for {cls.__name__}. Make sure you write a method that returns a serialzied version of this object."
    #     )


def filter_from_config(name: str, config: FilterConfig, log: Logger = getLogger()) -> Filter:
    """Creates a Filter from a config spefication. If no "filename" is present in the FilterConfig, it's
    assumed to be one of the default filtesr

    Parameters
    ----------
    name : str
        The unique name of a filter.
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
    filepath = config.get("filepath", None)

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
        for key, value in config.items():
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
    raise NotImplementedError(
        "Plugin from file has not been implemented! This method should take in a filepath and filter name, and return a runnable FilterPlugin!"
    )
