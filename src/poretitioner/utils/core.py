"""
===================
core.py
===================

Core classes and utilities that aren't specific to any part of the pipeline.

"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractclassmethod, abstractmethod
from collections import namedtuple
from dataclasses import Field, dataclass, fields
from os import PathLike
from pathlib import PosixPath
from typing import (
    AbstractSet,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    NewType,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import h5py
import numpy as np

from ..hdf5 import (
    HDF5_DatasetSerialableDataclass,
    HDF5_GroupSerialableDataclass,
    HDF5_GroupSerialiableDict,
)
from ..logger import Logger, getLogger
from .exceptions import CaptureSchemaVersionException
from .plugin import Plugin

__all__ = [
    "find_windows_below_threshold",
    "NumpyArrayLike",
    "ReadId",
    "PathLikeOrString",
    "ReadId",
    "Window",
    "WindowsByChannel",
]

# Generic wrapper type for array-like data. Normally we'd use numpy's arraylike type, but that won't be available until
# Numpy 1.21: https://stackoverflow.com/questions/40378427/numpy-formal-definition-of-array-like-objects

from ..hdf5 import NumpyArrayLike

# class NumpyArrayLike(np.ndarray):
#     def __new__(cls, signal: Union[np.ndarray, NumpyArrayLike]):
#         obj = np.copy(signal).view(
#             cls
#         )  # Optimization: Consider not making a copy, this is more error prone though: np.asarray(signal).view(cls)
#         return obj

#     def serialize_info(self, **kwargs) -> Dict:
#         """Creates a dictionary describing the signal and its attributes.

#         Returns
#         -------
#         Dict
#             A serialized set of attributes.
#         """
#         # When serializing, copy over any existing attributes already in self, and
#         # any that don't exist in self get taken from kwargs.
#         existing_info = self.__dict__
#         info = {key: getattr(self, key, kwargs.get(key)) for key in kwargs.keys()}
#         return {**info, **existing_info}

#     def deserialize_from_info(self, info: Dict):
#         """Sets attributes on an object from a serialized dict.

#         Parameters
#         ----------
#         info : Dict
#             Dictionary of attributes to set after deserialization.
#         """
#         for name, value in info.items():
#             setattr(self, name, value)

#     # Multiprocessing and Dask require pickling (i.e. serializing) their inputs.
#     # By default, this will drop all our custom class data.
#     # https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
#     def __reduce__(self):
#         reconstruct, arguments, object_state = super().__reduce__()
#         # Create a custom state to pass to __setstate__ when this object is deserialized.
#         info = self.serialize_info()
#         new_state = object_state + (info,)
#         # Return a tuple that replaces the parent's __setstate__ tuple with our own
#         return (reconstruct, arguments, new_state)

#     def __setstate__(self, state):
#         info = state[-1]
#         self.deserialize_from_info(info)
#         # Call the parent's __setstate__ with the other tuple elements.
#         super().__setstate__(state[0:-1])


# Unique identifier for a nanopore read.
ReadId = NewType("ReadId", str)

# Represents a path or a string representing a path. Having this union-type lets you use paths from Python's built-in pathib module (which is vastly superior to os.path)
PathLikeOrString = Union[str, PathLike]

####################################
###       Dataclass Helpers      ###
####################################


def dataclass_fieldnames(anything: Any) -> AbstractSet[str]:
    return types_by_field(anything).keys()


def types_by_field(anything: Any) -> Mapping[str, Type[Any]]:
    """Returns a mapping of an objects fields/attributes to the type of those fields.

    Parameters
    ----------
    object : Any
        Any object.

    Returns
    -------
    Dict[str, type]
        Mapping of an attribute name to its type.
    """
    types_by_field: Mapping[str, Type[type]] = {}
    if dataclasses.is_dataclass(anything):
        types_by_field = {field.name: field.type for field in dataclasses.fields(anything)}
    else:
        types_by_field = {name: type(attribute) for name, attribute in vars(anything).items()}
    return types_by_field


class Window(namedtuple("Window", ["start", "end"])):
    """Represents a general window of time.

    Parameters
    ----------
    start : float
        When this window starts.

    end : float
        When this window ends.
        End should always be greater than start.
    """

    @property
    def duration(self):
        """How long this window represents, measured by the difference between the start and end times.

        Returns
        -------
        float
            Duration of a window.

        Raises
        ------
        ValueError
            If window is invalid due to end time being smaller than start time.
        """
        if self.start > self.end:
            raise ValueError(
                f"Invalid window: end {self.end} is less than start {self.start}. Start should always be less than end."
            )

        duration = self.end - self.start
        return duration

    def overlaps(self, regions: List[Window]) -> List[Window]:
        """Finds all of the regions in the given list that overlap with the window.
        Needs to have at least one overlapping point; cannot be just adjacent.
        Incomplete overlaps are returned.

        Parameters
        ----------
        regions : List[Window] of windows.
            Start and end points to check for overlap with the specified window.
            All regions are assumed to be mutually exclusive and sorted in
            ascending order.

        Returns
        -------
        overlapping_regions : List[Window]
            Regions that overlap with the window in whole or part, returned in
            ascending order.

        Raises
        ------
        ValueError
            Somehow a region was neither before a window, after a window, or overlapping a window
            This can only happen if one of the region windows was invalid (e.g. has a start greater
            than end, violating locality).
        """

        window_start, window_end = self
        overlapping_regions = []
        for region in regions:
            region_start, region_end = region
            if region_start >= window_end:
                # Region comes after the window, we're done searching
                break
            elif region_end <= window_start:
                # Region ends before the window starts
                continue
            elif region_end > window_start or region_start >= window_start:
                # Overlapping
                overlapping_regions.append(region)
            else:
                e = f"Shouldn't have gotten here! Region: {region}, Window: {self}."
                raise ValueError(e)
        return overlapping_regions


def find_windows_below_threshold(
    time_series: NumpyArrayLike, threshold: float | int
) -> List[Window]:
    """
    Find regions where the time series data points drop at or below the
    specified threshold.

    Parameters
    ----------
    time_series : NumpyArrayLike
        Array containing time series data points.
    threshold : float or int
        Find regions where the time series drops at or below this value

    Returns
    -------
    list of Windows (start, end): List[Window]
        Each item in the list represents the (start, end) points of regions
        where the input array drops at or below the threshold.
    """

    diff_points = np.where(np.abs(np.diff(np.where(time_series <= threshold, 1, 0))) == 1)[0]
    if time_series[0] <= threshold:
        diff_points = np.hstack([[0], diff_points])
    if time_series[-1] <= threshold:
        diff_points = np.hstack([diff_points, [time_series.duration]])
    return [Window(start, end) for start, end in zip(diff_points[::2], diff_points[1::2])]


def channel_name(channel_number: int) -> str:
    """Gets the channel name associated with a channel number, e.g. 'Channel_19'.

    Parameters
    ----------
    channel_number : int
        Which channel to generate the name for.

    Returns
    -------
    str
        The name of the channel as a string.
    """
    return f"Channel_{channel_number}"


class WindowsByChannel(HDF5_GroupSerialiableDict[str, NumpyArrayLike]):
    def __init__(self, windows_by_channel: Mapping[int, List[Window]], *args):
        numpy_array_mapping = {
            channel_name(channel_number): NumpyArrayLike(windows)
            for channel_number, windows in windows_by_channel.items()
        }
        super().__init__(numpy_array_mapping, *args)

    def name(self) -> str:
        return "capture_windows"


def stripped_by_keys(dictionary: Optional[Dict], keys_to_keep: Iterable) -> Dict:
    """Returns a dictionary containing keys and values from dictionary,
    but only keeping the keys in `keys_to_keep`.

    Returns empty dict if `dictionary` is None.

    Parameters
    ----------
    dictionary : Optional[Dict]
        Dictionary to strip down by keys.
    keys_to_keep : Iterable
        Which keys of the dictionary to keep.

    Returns
    -------
    Dict
        Dictionary containing only keys from `keys_to_keep`.
    """
    dictionary = {} if dictionary is None else dictionary
    dictionary = {key: value for key, value in dictionary.items() if key in keys_to_keep}
    return dictionary
