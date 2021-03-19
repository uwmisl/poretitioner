"""
===================
core.py
===================

Core classes and utilities that aren't specific to any part of the pipeline.

"""
from __future__ import annotations
from abc import ABC, abstractclassmethod, abstractmethod

from collections import namedtuple
from dataclasses import dataclass, Field, fields
import dataclasses
from os import PathLike
from typing import Any, Dict, Iterable, List, Mapping, Optional, NewType, Protocol, Union, Sequence, Type
from h5py import File as Fast5File
import h5py
from .exceptions import HDF5SerializationException, HDF5GroupSerializationException, CaptureSchemaVersionException
from pathlib import PosixPath
from .plugin import Plugin
from ..logger import Logger, getLogger

import numpy as np

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

class NumpyArrayLike(np.ndarray):
    def __new__(cls, signal: NumpyArrayLike):
        obj = np.copy(signal).view(
            cls
        )  # Optimization: Consider not making a copy, this is more error prone though: np.asarray(signal).view(cls)
        return obj

    def serialize_info(self, **kwargs) -> Dict:
        """Creates a dictionary describing the signal and its attributes.

        Returns
        -------
        Dict
            A serialized set of attributes.
        """
        # When serializing, copy over any existing attributes already in self, and
        # any that don't exist in self get taken from kwargs.
        existing_info = self.__dict__
        info = {key: getattr(self, key, kwargs.get(key)) for key in kwargs.keys()}
        return {**info, **existing_info}

    def deserialize_from_info(self, info: Dict):
        """Sets attributes on an object from a serialized dict.

        Parameters
        ----------
        info : Dict
            Dictionary of attributes to set after deserialization.
        """
        for name, value in info.items():
            setattr(self, name, value)

    # Multiprocessing and Dask require pickling (i.e. serializing) their inputs.
    # By default, this will drop all our custom class data.
    # https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
    def __reduce__(self):
        reconstruct, arguments, object_state = super().__reduce__()
        # Create a custom state to pass to __setstate__ when this object is deserialized.
        info = self.serialize_info()
        new_state = object_state + (info,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (reconstruct, arguments, new_state)

    def __setstate__(self, state):
        info = state[-1]
        self.deserialize_from_info(info)
        # Call the parent's __setstate__ with the other tuple elements.
        super().__setstate__(state[0:-1])



# Unique identifier for a nanopore read.
ReadId = NewType("ReadId", str)

# Represents a path or a string representing a path. Having this union-type lets you use paths from Python's built-in pathib module (which is vastly superior to os.path)
PathLikeOrString = Union[str, PathLike]

####################################
###       Dataclass Helpers      ###
####################################

def types_by_field(object: Any) -> Mapping[str, Type[Any]]:
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
    types_by_field: Mapping[str, Type[Any]] = {}
    if dataclasses.is_dataclass(object):
        types_by_field = {field.name:field.type for field in dataclasses.fields(object)}
    else:
        types_by_field = {name: type(attribute) for name, attribute in vars(object).items()}
    return types_by_field


####################################
###         Fast5 Helpers        ###
####################################

def hdf5_dtype(object: Any) -> Optional[np.dtype]:
    """Returns the proper h5py dtype for an object, if one is necessary.
    Otherwise returns None.

    For us, this is mostly needed in the case of storing numpy data or string data,

    since numpy data has a specific dtype, and strings have a variable length and an assumed encoding (e.g. "utf-8")

    For more info on how h5py handles strings, see [1, 2].

    [1] - https://docs.h5py.org/en/stable/strings.html#strings
    [2] - https://docs.h5py.org/en/stable/special.html?highlight=string_dtype#variable-length-strings

    Parameters
    ----------
    object : Any
        Some object you want the dtype for if it's necessary, but are fine not having one
        if it's not.

    Returns
    -------
    Optional[np.dtype]
        The numpy datatype for an object if it has one, or if it's a string, and None otherwise.
    """
    if isinstance(object, str):
        return h5py.string_dtype(length=len(object))
    elif hasattr(object, "dtype"):
         # Is this already a numpy-like object with a dtype? If so, just use that.
        return object.dtype
    return None # For most cases, h5py can determine the dtype from the data itself.

class HasFast5(Protocol):
    f5: Fast5File

# Note: We never create or instantiate AttributeManagers directly, instead we borrow its interface.
#       3 Laws to keep in mind with Attributes:
#
#
#    1) They may be created from any scalar or NumPy array
#
#    2) Each attribute should be small (generally < 64k)
#
#    3) There is no partial I/O (i.e. slicing); the entire attribute must be read.
#
#       https://docs.h5py.org/en/stable/high/attr.html

HDF5_Dataset = NewType("HDF5_Dataset", h5py.Dataset)
HDF5_Group = NewType("HDF5_Group", h5py.Group)
HDF5_Attribute = NewType("HDF5_Attribute", h5py.AttributeManager)

HDF5_Type = Union[HDF5_Attribute, HDF5_Dataset, HDF5_Group]

class HDF5Serializing(ABC, Plugin):
    """Any object that can be HDFSserialized.

    A plugin is just a class with a name :)

    Don't instantiate this directly, rather subclass.
    """
    @classmethod
    @abstractmethod
    def from_a(cls, a: HDF5_Type, log: Optional[Logger] = None) -> HDF5Serializing:
        """Creates an instance of this class (from) (a) HDF5_Type.

        Parameters
        ----------
        a : HDF5_Types
            Instance of an HDF5Type (e.g. a h5py.Group).

        log : Logger, optional
            Logger to use for information/warnings/debug

        Returns
        -------
        HDF5Serializing
            An instance of this class with data derived from (a) HDF5_Type.

        Raises
        ------
        NotImplementedError
            This method wasn't implemented, but needs to be.
        """
        raise NotImplementedError(f"{cls!s} is missing an implementation for {HDF5Serializing.from_a.__name__}")

    @abstractmethod
    def as_a(self, a: HDF5_Type, log: Optional[Logger] = None) -> HDF5_Type:
        """Returns this object, formatted (as) (a) given HDF5 type (thus the name).

        Parameters
        ----------
        a : HDF5_Types
            One of the HDF5 types we understand.

        log : Logger, optional
            Logger to use for information/warnings/debug

        Returns
        -------
        HDF5_Type
            This object serialized to a given HDF5 type.

        Raises
        ------
        NotImplementedError
            This method wasn't implemented, but needs to be.
        """
        raise NotImplementedError(f"{self!s} is missing an implementation for {HDF5Serializing.as_a.__name__}!")

class HDF5AttributeSerializing(HDF5Serializing):
    @classmethod
    def name(cls) -> str:
        """Attribute name that this object will be stored under.
        i.e. If this method returns "KarlMarx", then a subsequent call to

        `self.as_attribute(Group("/Foo/bar/"))`

        Will result in an attribute 'KarlMarx' at /Foo/bar

        Be double-sure to override this if you want it to be anything other than the class name.

        Returns
        -------
        str
            Name to use in the Fast5 file.
        """
        return cls.__name__

    @abstractmethod
    def from_a(cls, a: HDF5_Attribute, log: Optional[Logger] = None) -> HDF5AttributeSerializing:
        """Creates an instance of this class (from) (a) HDF5_Type.

        Parameters
        ----------
        a : HDF5_Types
            Instance of an HDF5Type (e.g. a h5py.Group).

        log : Logger, optional
            Logger to use for information/warnings/debug

        Returns
        -------
        HDF5Serializing
            An instance of this class with data derived from (a) HDF5_Type.

        Raises
        ------
        NotImplementedError
            This method wasn't implemented, but needs to be.
        """
        raise NotImplementedError(f"{cls!s} is missing an implementation for {HDF5Serializing.from_a.__name__}")

    @abstractmethod
    def as_a(self, a: HDF5_Attribute, log: Optional[Logger] = None) -> HDF5_Attribute:
        """Returns this object, formatted (as) (a) HDF5_Attribute.

        Parameters
        ----------
        a : HDF5_Attribute
            An HDF5 attribute types we understand.

        log : Logger, optional
            Logger to use for information/warnings/debug

        Returns
        -------
        HDF5_Type
            This object serialized to a given HDF5 type.

        Raises
        ------
        NotImplementedError
            This method wasn't implemented, but needs to be.
        """
        raise NotImplementedError(f"{self!s} is missing an implementation for {HDF5Serializing.as_a.__name__}!")


class HDF5GroupSerializing(HDF5Serializing):
    """Objects adhering to the `HDF5GroupSerializable` can be written to and
    read directly from hd5 Groups.
    """
    @classmethod
    def name(cls) -> str:
        """Group name that this object will be stored under.
        i.e. If this method returns "patrice_lmb", then a subsequent call to

        `self.as_group(Group("/Foo/bar/"))`

        Will return a group at /Foo/bar/patrice_lmb

        Be double-sure to override this if you want it to be anything other than the class name.

        Returns
        -------
        str
            Name to use in the Fast5 file.
        """
        return cls.__name__

    def as_group(self, parent_group: h5py.Group, log: Optional[Logger] = None) -> h5py.Group:
        """Stores and Returns this object as an HDF5 Group, rooted at the group passed in.
        This should be useable to directly set the contents of an Hdf5 group.
        This method should also create the group named 'name' in the parent_group, if it doesn't already exist.

        class Baz(HDF5GroupSerializable):
            def name(self):
                return "boop"
            # ...Implementation

        my_hdf5_file = h5py.File("/path/to/file")
        foo_group = filts.require_group("/foo")

        my_serial = Baz()
        baz_group = foo_group.require_group(my_serial.name()) # Make space in the file for Baz at f'/foo/{my_serial.name()}'
        my_serialized_group = my_serial.as_group(foo_group) # Sets "/foo/boop" group to the serialized group

        my_serialized_group # /boop group, rooted at /foo/

        Parameters
        ----------
        parent_group : h5py.Group
            Which group to store this group under. This doesn't necessarily have to be the root group of the file.

        Returns
        -------
        h5py.Group
            Group that stores a serialization of this instance.
        """
        ...

    @classmethod
    def from_group(cls, group: h5py.Group, log: Optional[Logger] = None) -> HDF5GroupSerializable:
        """Serializes this object FROM an HDF5 Group.

        class Baz(HDF5GroupSerializable):
            # ...Implementation

        my_hdf5_file = h5py.File("/path/to/file")
        baz_serialized_group = filts.require_group("/baz")

        baz = Baz.from_group(baz_serialized_group) # I now have an instance of Baz.

        Parameters
        ----------
        group : h5py.Group
            HDF5 Group that can be serialized into this instance.

        Returns
        -------
        HDF5GroupSerializable
            Instance of an adherent to this protocol.
        """
        ...


class HDF5GroupSerializable(HDF5GroupSerializing):
    """Base class for objects that can be written to and
    read directly from hd5 Groups.

    Not meant to be instantiated directly. Instead, subclass and make sure your
    `as_group` implementation uses the group created by `super().as_group(...)`.

    NOTE: Make sure to call super().as_group(...)
    """

    @classmethod
    def name(cls) -> str:
        """Group name that this object will be stored under.
        i.e. If this method returns "patrice_lmb", then a subsequent call to

        `self.as_group(Group("/Foo/bar/"))`

        Will return a group at /Foo/bar/patrice_lmb

        Override this if you want it to be anything other than the class name.

        Returns
        -------
        str
            Name to use in the Fast5 file.
        """
        return cls.__name__

    def as_group(self, parent_group: h5py.Group, log: Optional[Logger] = None) -> h5py.Group:
        serialized_group = parent_group.require_group(self.name())
        # Note: This does nothing but register a group with the name 'name' in the parent group.
        #       Implementers must now write their serialized instance to this group.
        return serialized_group

    @classmethod
    @abstractmethod
    def from_group(cls, group: h5py.Group, log: Optional[Logger] = None) -> HDF5GroupSerializable:
        raise NotImplementedError(f"from_group not implemented for {cls.__name__}. Make sure you write a method that returns a serialzied version of this object.")

    @classmethod
    def from_a(cls, a: HDF5_Type, log: Logger) -> HDF5Serializing:
        return cls.from_group(a, log=log)

    def as_a(self, a: HDF5_Type, log: Logger) -> HDF5_Type:
        return self.as_group(a, log=log)


import importlib

def get_class_for_name(name: str, module_name: str = __name__) -> Type:
    """Gets a class from a module based on its name.
    Tread carefully with this. Personally I feel like it's only safe to use
    with dataclasses with known interfaces.

    Parameters
    ----------
    name : str
        Name of the class we're trying to get the class object for.

    module_name: str, optional
        Which module to get a class from, by defualt __name__.

    Returns
    -------
    Type
        [description]
    """
    this_module = importlib.import_module(module_name)
    this_class = getattr(this_module, name)
    return this_class

class DataclassHDF5GroupSerialable(HDF5GroupSerializable):

    def as_group(self, parent_group: h5py.Group, log: Optional[Logger] = None) -> h5py.Group:
        log = log if log is not None else getLogger()

        """Returns this object as an HDF5 Group."""
        my_group = super().as_group(parent_group)

        for field_name, value in dataclass.asdict(self).items():
            if isinstance(value, HDF5GroupSerializable):
                # This value is actually its own group.
                # So we create a new group rooted at our dataclass's group
                # And assign it the value of whatever the group of the value is.
                parent_group.require_group(field_name)
                field_group = value.as_group(field_name, log=log)
            elif isinstance(value, HDF5Serializing):
                my_group.attrs.create(field_name, value, dtype=hdf5_dtype(value))
            else:
                my_group.attrs.create(field_name, value, dtype=hdf5_dtype(value))
        return my_group

    @classmethod
    def from_group(cls, group: h5py.Group, log: Optional[Logger] = None) -> DataclassHDF5GroupSerialable:
        log = log if log is not None else getLogger()
        if not log:
            log = getLogger()
        my_instance = cls.__new__(cls)

        # First, copy over attrs:
        for name, value in group.attrs.items():
            object.__setattr__(my_instance, name, value)

        # Then, copy over any datasets or groups.
        for name, value in group.items():
            if isinstance(value, h5py.Dataset):
                # Assuming we're storing a numpy array as this dataset
                buffer = np.empty(value.shape, dtype=value.dtype)
                # Copies the values into our buffer
                value.read_direct(buffer)
                object.__setattr__(my_instance, name, buffer)
            elif isinstance(value, h5py.Group):
                # If it's a group, we have to do a little more work
                # 1) Find the class described by the group
                #   1.1) Verify that we actually know a class by that name. Raise an exception if we don't.
                #   1.2) Verify that that class has a method to create an instance group a group.
                # 2) Create a new class instance from that group
                # 3) Set this object's 'name' field to the object we just created.
                try:
                    ThisClass = get_class_for_name(name)
                except AttributeError as e:
                    serial_exception = HDF5GroupSerializationException(f"We couldn't serialize group named {name} (group is attached in the exception.", e, group=value)
                    log.exception(serial_exception.msg, serial_exception)
                    raise serial_exception

                #assert get_class_for_name(name) and isinstance(), f"No class found that corresponds to group {name}! Make sure there's a corresponding dataclass named {name} in this module scope!"

                try:
                    this_instance = ThisClass.from_group(value, log=log)
                except AttributeError as e:
                    serial_exception = HDF5GroupSerializationException(f"We couldn't serialize group named {name!s} from class {ThisClass!s}. It appears {ThisClass!s} doesn't implement the {HDF5GroupSerializing.__name__} protocol. Group is attached in the exception.", e, group=value)
                    log.exception(serial_exception.msg, serial_exception)
                    raise serial_exception

                object.__setattr__(my_instance, name, this_instance)

        return my_instance


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

    # Katie Q: I still don't understand how this function works haha. Let's talk next standup?
    diff_points = np.where(
        np.abs(np.diff(np.where(time_series <= threshold, 1, 0))) == 1
    )[0]
    if time_series[0] <= threshold:
        diff_points = np.hstack([[0], diff_points])
    if time_series[-1] <= threshold:
        diff_points = np.hstack([diff_points, [time_series.duration]])
    return [
        Window(start, end) for start, end in zip(diff_points[::2], diff_points[1::2])
    ]


@dataclass
class WindowsByChannel:
    by_channel: Dict[int, List[Window]]

    def __init__(self, *args):
        self.by_channel = dict()

    def __getitem__(self, channel_number: int):
        return self.by_channel[channel_number]

    def __setitem__(self, channel_number: int, windows: List[Window]):
        self.by_channel[channel_number] = windows

    def keys(self):
        return self.by_channel.keys()


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
    dictionary = {
        key: value for key, value in dictionary.items() if key in keys_to_keep
    }
    return dictionary
