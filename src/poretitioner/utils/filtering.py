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
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np

from ..logger import Logger, getLogger
from ..signals import Capture
from .configuration import FilterConfig
from .core import NumpyArrayLike, PathLikeOrString

CaptureOrTimeSeries = Union[Capture, NumpyArrayLike]


@dataclass(frozen=True)
class PATH:
    FILTER = f"/Filter/"

    @classmethod
    def filter_path_for_name(cls, name: str) -> str:
        filter_path = str(PosixPath(PATH.FILTER, name))
        return filter_path

    @classmethod
    def filter_pass_path_for_read_id(cls, read_id: str) -> str:
        pass_path = str(PosixPath(PATH.FILTER, "pass", read_id))
        return pass_path


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
    def __init__(
        self, minimum: Optional[float] = None, maximum: Optional[float] = None
    ):
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
    """Filters for captures with standard deviations in some range."""

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
    """Filters for captures with an arithmetic mean within a range."""

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
    """Filters for captures with a median within a range."""
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
    """Filters for captures with a minimum within a range."""
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
    """Filters for captures with a maximum within a range."""
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
    """Filters captures based on their length."""

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


def apply_feature_filters(
    capture: CaptureOrTimeSeries, filters: List[FilterPlugin]
) -> bool:
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


def apply_filters_to_read(config, f5, read_id, filter_name):
    passed_filters = True

    # Check whether the capture was ejected
    if "ejected" in config["filters"][filter_name]:
        only_use_ejected_captures = config["filters"][filter_name]["ejected"]  # TODO
        if only_use_ejected_captures:
            capture_ejected = check_capture_ejection_by_read(f5, read_id)
            if not capture_ejected:
                passed_filters = False
                return passed_filters
    else:
        only_use_ejected_captures = False  # could skip this, leaving to help read logic

    # Apply all the filters
    get_raw_signal()
    signal = raw_signal_utils.get_fractional_blockage_for_read(f5, read_id)
    # print(config["filters"][filter_name])
    # print(f"min = {np.min(signal)}")
    passed_filters = apply_feature_filters(signal, config["filters"][filter_name])
    return passed_filters


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


def filter_like_existing(
    config, example_fast5, example_filter_path, fast5_files, new_filter_path
):
    # Filters a set of fast5 files exactly the same as an existing filter
    # TODO : #68 : implement
    raise NotImplementedError()


def get_filter_pass_path(read_id):
    path = str(PosixPath(PATH.FILTER, "pass", read_id))
    return path


DEFAULT_PLUGINS = [
    RangeFilter,
    MeanFilter,
    StandardDeviationFilter,
    MedianFilter,
    MinimumFilter,
    MaximumFilter,
    LengthFilter,
]


def plugin_from_config(config: FilterConfig, log: Logger = getLogger()) -> FilterPlugin:
    """[summary]

    Parameters
    ----------
    config : FilterConfig
        Filter configuration to build the plugin.
    log : Logger, optional
        Logger to use for information/warnings/debug, by default getLogger()

    Returns
    -------
    FilterPlugin
        [description]

    Raises
    ------
    AttributeError
        A filter plugin could not be built from the configuration description. If this error is raised, be sure to check
        1) A plugin class with the name in the configuration is defined at the filepath described in the configuration
        2) The plugin class inherits from the `FilterPlugin` abstract base class.
    """
    name = config.name
    filepath = config.filepath
    attributes: Dict[str, Any] = config.attributes

    # TODO: For non-default FilterPlugins, load/unpickle the class from the filepath. https://github.com/uwmisl/poretitioner/issues/91
    plugin = None

    default_plugins = {plugin.name: plugin for plugin in DEFAULT_PLUGINS}
    if name in default_plugins:
        plugin = default_plugins[name]()
    else:
        # TODO: For non-default FilterPlugins, load/unpickle the class from the filepath. https://github.com/uwmisl/poretitioner/issues/91
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

    return plugin


def plugin_from_file(name: str, filepath: PathLikeOrString):
    # TODO: For non-default FilterPlugins, load/unpickle the class from the filepath. https://github.com/uwmisl/poretitioner/issues/91
    pass


def get_plugins(filter_configs: List[FilterConfig]) -> List[FilterPlugin]:
    """Creates FilterPlugins from a list of filter configurations.

    Parameters
    ----------
    filter_configs : List[FilterConfig]
        [description]

    Returns
    -------
    List[FilterPlugin]
        [description]
    """
    plugins = [plugin_from_config(config) for config in filter_configs]
    return plugins
