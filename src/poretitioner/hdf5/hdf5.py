"""
===================
hdf5.py
===================

The Hierarchical Data Format version 5 (HDF5) defines a
a file format for storing and organzing massive amounts of
hiearchical data.

This module attempts to encapsulate the rich features of HDF5
alongside your favorite python3.7+ constructs
(e.g dataclasses)

[1] - https://en.wikipedia.org/wiki/Hierarchical_Data_Format

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger, getLogger
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

import h5py
import numpy as np
from h5py import File as Fast5File
from h5py._hl.base import Empty

from .exceptions import HDF5_GroupSerializationException, HDF5_SerializationException

# Generics
T = TypeVar("T")
S = TypeVar("S")


class NumpyArrayLike(np.ndarray):
    """This class represents a numpy array with extra attributes and functionality.

    Subclasses of NumpyArrayLike can be treated exactly like numpy arrays computationally

    By default, we serialize class attributes alone.

    For more fine-grained control over what information is stored during serialization/pickling,
    implementers should override the `serialize_info` `deserialize_from_info`

    """

    def __new__(cls, data: Union[np.ndarray, NumpyArrayLike]):
        obj = np.copy(data).view(
            cls
        )  # Optimization: Consider not making a copy, this is more error prone though: np.asarray(data).view(cls)
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


####################################
###         Fast5 Helpers        ###
####################################


# NOTE: Sadly, many of these can't be used until "RuntimeError: Unable to create attribute (object header message is too large)" https://github.com/h5py/h5py/issues/1855
# The goal is to provide high-level data-class representations of HDF5 objects, so users can just describe their structures as python dataclasses instead of finagling with h5py. 
# Unfortunately, there's currently a bug "RuntimeError: Unable to create attribute (object header message is too large)" that can only be fixed by delving into the low-level API ()

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
    return None  # For most cases, h5py can determine the dtype from the data itself.


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
    import importlib

    this_module = importlib.import_module(module_name)
    this_class = getattr(this_module, name)
    return this_class


class HasFast5(Protocol):
    f5: Union[Fast5File, HDF5_Group]


###########################################################
#
#                      HDF5 Base Types
#
###########################################################

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

# Attrs are really just mappings from names to data/objects.
HDF5_Attribute_Objects = Mapping[str, Optional[Any]]


class IsAttr(Protocol):
    """A special protocol for objects that are just meant to be set data attributes, and don't
    need any special HDF5 consdiration (e.g. a class that just needs to store a few numbers).
    """

    def as_attr(self) -> np.dtype:
        ...

    def from_attr(self, attr) -> IsAttr:
        ...


class HDF5IsAttr(IsAttr):
    def as_attr(self) -> np.dtype:
        ...

    def from_attr(self, attr) -> IsAttr:
        ...


class HasAttrs(Protocol):
    def get_attrs(self) -> HDF5_Attributes:
        ...

    def create_attr(self, name: str, value: Optional[Any], log: Optional[Logger] = None):
        """Adds an attribute to the current object.

        Any existing attribute with this name will be overwritten.

        Parameters
        ----------
        name : str
            Name of the attribute.
        value : Optional[Any]
            Value of the attribute.
        """
        ...

    def create_attrs(self, attrs: HDF5_Attributes, log: Optional[Logger] = None):
        """Adds multiple attributes to the current object.

        Any existing attribute with the names in attrs will be overwritten.

        Parameters
        ----------
        attrs :
            Name of the attribute.
        value : Optional[Any]
            Value of the attribute.
        """

    def object_from_attr(self, name: str, log: Optional[Logger] = None) -> Optional[Any]:
        """Creates an object from an attribute (if one could be made).
        # TODO: Plugin Register via Plugins

        Parameters
        ----------
        name : str
            Name of the attribute.

        Returns
        ----------
        An instantiated object represented by this attr, or None if one couldn't be found.
        """
        ...

    def objects_from_attrs(
        self, attrs: HDF5_Attributes, log: Optional[Logger] = None
    ) -> HDF5_Attribute_Objects:
        """Creates mapping of attribute names to their serialzed objects (if one could be made).

        Parameters
        ----------
        name : str
            Name of the attribute.

        Returns
        ----------
        An instantiated object represented by this attr, or None if one couldn't be found.
        """
        ...


class HDF5_AttributeHaving(HasAttrs):
    def __init__(self, has_attrs: Optional[HasAttrs]):
        super().__init__()
        self.attrs = self.get_attrs() if has_attrs is None else has_attrs.get_attrs()

    def get_attrs(self) -> HDF5_Attributes:
        return self.attrs

    def create_attr(self, name: str, value: Optional[Any], log: Optional[Logger] = None):
        """Adds an attribute to the current object.

        WARNING: Any existing attribute will be overwritten!

        This method will coerce value to a special 'Empty' type used by HDF5 if the value
        provided is zero-length or None. For more on Attributes and Empty types, see [1, 2]

        [1] - https://docs.h5py.org/en/stable/high/attr.html#attributes
        [2] - https://docs.h5py.org/en/stable/high/dataset.html?highlight=Empty#creating-and-reading-empty-or-null-datasets-and-attributes

        Parameters
        ----------
        name : str
            Name of the attribute.
        value : Optional[Any]
            Value of the attribute. This method will coerce this value
            to a special Empty object if it's zero-length or None [2].
        """

        if value is None or value == "" or (hasattr(value, "__len__") and len(value) < 1):
            empty = h5py.Empty(dtype=np.uint8)
            self.get_attrs().create(name, empty)
        elif isinstance(value, HDF5IsAttr):
            attr_value = value.as_attr()
            self.get_attrs().create(name, value, dtype=hdf5_dtype(attr_value))
        else:
            self.get_attrs().create(name, value, dtype=hdf5_dtype(value))

    def create_attrs(self, attrs: HDF5_Attributes, log: Optional[Logger] = None):
        for attr_name, attr_value in attrs.items():
            self.create_attr(attr_name, attr_value, log=log)

    def object_from_attr(self, name: str, log: Optional[Logger] = None) -> Optional[Any]:
        log = log if log is not None else getLogger()
        try:
            attr_value = self.get_attrs()[name]
        except AttributeError:
            log.warning(
                f"Could not find an attribute with the name '{name}' on object {self!r}. Returning None"
            )
            return None

        if attr_value.shape is None:
            """
            From the Docs:

            An empty dataset has shape defined as None,
            which is the best way of determining whether a dataset is empty or not.
            An empty dataset can be “read” in a similar way to scalar datasets.

            [1] - https://docs.h5py.org/en/stable/high/dataset.html?highlight=Empty#creating-and-reading-empty-or-null-datasets-and-attributes
            """
            return ""
        return bytes.decode(bytes(attr_value), encoding="utf-8")

    def objects_from_attrs(self, log: Optional[Logger] = None) -> HDF5_Attribute_Objects:
        objects: HDF5_Attribute_Objects = {
            attr_name: self.object_from_attr(attr_name, log=log)
            for attr_name in self.get_attrs().keys()
        }
        return objects

    def copy_attr(self, name: str, source: HDF5_AttributeHaving):
        """Copy a single attribute from a source.
        This will overwrite any attribute of this name, if one exists.

        Parameters
        ----------
        name : str
            Which attribute to copy.
        from : HDF5_AttributeHaving
            Which attribute-haver to copy from.
        """
        self.create_attr(name, source.get_attrs()[name])

    def copy_all_attrs(self, source: HDF5_AttributeHaving):
        """Copy a all attributes from a source.
        This will overwrite any attributes sharing the same names, if any exists.

        Parameters
        ----------
        from : HDF5_AttributeHaving
            Which attribute-haver to copy all attributes from.
        """
        for name in source.get_attrs().keys():
            self.copy_attr(name, source)


class HDF5_ParentHaving:
    @property
    def parent(self) -> HDF5_Group:
        return HDF5_Group(self.parent)


class HDF5_Dataset(h5py.Dataset, NumpyArrayLike, HDF5_AttributeHaving, HDF5_ParentHaving):

    def __new__(cls, dataset: NumpyArrayLike) -> HDF5_Dataset:
        if isinstance(dataset, HDF5_Dataset):
            return dataset
        
        self = dataset
        

    def __init__(self, dataset: h5py.Dataset):
        self._dataset = dataset

    def __getattr__(self, attrib: str):
        return getattr(self._dataset, attrib)


class HDF5_Group(h5py.Group, HDF5_AttributeHaving, HDF5_ParentHaving):
    def __new__(cls, group: Optional[h5py.Group]) -> HDF5_Group:
        if isinstance(group, HDF5_Group):
            return group
        hdf5_group = super().__new__(cls, group)
        hdf5_group._group = group
        return hdf5_group

    def __init__(self, group: Optional[h5py.Group]):
        if isinstance(group, HDF5_Group):
            return
        super().__init__(group.id)
        self._group = group

    @property
    def parent(self) -> HDF5_Group:
        return HDF5_Group(self._group.parent)

    def require_group(self, name: str):
        return HDF5_Group(self._group.require_group(name))

    def require_dataset(self, name, data, dtype, shape, **kwds):
        return HDF5_Dataset(self._group.require_dataset(name, shape, data=data, dtype=dtype,**kwds))

    def __getattr__(self, attrib: str):
        return getattr(self._group, attrib)


class HDF5_Attributes(h5py.AttributeManager, HDF5_ParentHaving):
    def __init__(self, attrs: h5py.AttributeManager):
        self.attrs = attrs

    def __getattr__(self, attrib: str):
        return getattr(self.attrs, attrib)


HDF5_Type = Union[HDF5_Dataset, HDF5_Group, HDF5_Attributes]


class HDF5_Serializing(ABC):
    """Any object that can be HDFSserialized.

    Don't instantiate this directly, rather subclass.
    """

    @classmethod
    @abstractmethod
    def from_a(cls, a: HDF5_Type, log: Optional[Logger] = None) -> HDF5_Serializing:
        """Creates an instance of this class (from) (a) HDF5_Type.

        Parameters
        ----------
        a : HDF5_Types
            Instance of an HDF5Type (e.g. a h5py.Group).

        log : Logger, optional
            Logger to use for information/warnings/debug

        Returns
        -------
        HDF5_Serializing
            An instance of this class with data derived from (a) HDF5_Type.

        Raises
        ------
        NotImplementedError
            This method wasn't implemented, but needs to be.
        """
        raise NotImplementedError(
            f"{cls!s} is missing an implementation for {HDF5_Serializing.from_a.__name__}"
        )

    @abstractmethod
    def as_a(
        self, a: HDF5_Type, parent: Optional[HDF5_Group] = None, log: Optional[Logger] = None
    ) -> HDF5_Type:
        """Returns this object, formatted (as) (a) given HDF5 type (thus the name).

        Parameters
        ----------
        a : HDF5_Types
            One of the HDF5 types we understand.

        parent : Optional[HDF5_Group]
            The parent group to which this object should be added/updated.
            If parent is None, we return the HDFS_Type proxy in isolation (e.g)
            not attached to the parent if it doesn't already have one. None by default.

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
        raise NotImplementedError(
            f"{self!s} is missing an implementation for {HDF5_Serializing.as_a.__name__}!"
        )

    @abstractmethod
    def update(self, log: Optional[Logger] = None):
        """Makes sure any changes have been reflected in the underlying object.

        Parameters
        ----------
        log : Optional[Logger], optional
            Logger to use, by default None

        Raises
        ------
        NotImplementedError
            This method wasn't implemented.
        """
        raise NotImplementedError(
            f"{self!s} is missing an implementation for {HDF5_Serializing.update.__name__}!"
        )


###########################################################
#
#                      HDF5 Groups
#
###########################################################


class HDF5_GroupSerializing(HDF5_Serializing, HDF5_AttributeHaving):
    """Objects adhering to the `HDF5_GroupSerializable` can be written to and
    read directly from hd5 Groups.
    """

    def name(self) -> str:
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
        return self.__class__.__name__

    def as_group(self, parent_group: HDF5_Group, log: Optional[Logger] = None) -> HDF5_Group:
        """Stores and Returns this object as an HDF5 Group, rooted at the group passed in.
        This should be used to directly set the contents of an Hdf5 group.
        This method should also create the group named 'name' in the parent_group, if it doesn't already exist.

        class Baz(HDF5_GroupSerializable):
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
    def from_group(cls, group: HDF5_Group, log: Optional[Logger] = None) -> HDF5_GroupSerializable:
        """Serializes this object FROM an HDF5 Group.

        class Baz(HDF5_GroupSerializable):
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
        HDF5_GroupSerializable
            Instance of an adherent to this protocol.
        """
        ...


class HDF5_GroupSerializable(HDF5_GroupSerializing):
    """Base class for objects that can be written to and
    read directly from hd5 Groups.

    Not meant to be instantiated directly. Instead, subclass and make sure your
    `as_group` implementation uses the group created by `super().as_group(...)`.

    NOTE: Make sure to call super().as_group(...)
    """

    def name(self) -> str:
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
        return self.__class__.__name__

    def as_group(self, parent_group: HDF5_Group, log: Optional[Logger] = None) -> HDF5_Group:
        new_group = parent_group.require_group(self.name())
        parent_group[self.name()] = self
        # Note: This does nothing but register a group with the name 'name' in the parent group.
        #       Implementers must now write their serialized instance to this group.
        return self

    @classmethod
    @abstractmethod
    def from_group(cls, group: HDF5_Group, log: Optional[Logger] = None) -> HDF5_GroupSerializable:
        raise NotImplementedError(
            f"from_group not implemented for {cls.__name__}. Make sure you write a method that returns a serialzied version of this object."
        )

    def require_group_from_group(
        self, parent_group: HDF5_Group, log: Optional[Logger] = None
    ) -> HDF5_GroupSerializable:
        # child_group = parent_group.require_group(self.name())
        child_group = self.as_group(parent_group, log=log)

    @classmethod
    def from_a(cls, a: HDF5_Group, log: Logger) -> HDF5_Serializing:
        return cls.from_group(parent_group=a, log=log)

    def as_a(self, a: HDF5_Type, log: Logger) -> HDF5_Type:
        return self.as_group(parent_group=a, log=log)

    def update(self, log: Optional[Logger] = None):
        self.as_a(self._group.parent, log=log)


class HDF5_GroupSerialiableDict(Dict[T, S], HDF5_GroupSerializable):
    def as_group(self, parent_group: HDF5_Group, log: Optional[Logger] = None) -> HDF5_Group:
        log = log if log is not None else getLogger()

        my_group = parent_group.require_group(self.name())
        """Returns this object as an HDF5 Group."""
        for field_name, field_value in self.items():
            if isinstance(field_value, HDF5_GroupSerializable):
                # This value is actually its own group.
                # So we create a new group rooted at our dataclass's group
                # And assign it the value of whatever the group of the value is.
                field_value.as_a(my_group, log=log)
                # my_group.require_group(field_name)
                # my_group[field_name] = field_value
            elif isinstance(field_value, HDF5_DatasetSerializable):
                field_value.as_a(parent_group, log)
            else:
                my_group.create_attr(field_name, field_value)
        return my_group

    @classmethod
    def from_group(
        cls, group: HDF5_Group, log: Optional[Logger] = None
    ) -> HDF5_GroupSerialableDataclass:
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
                object.__setattr__(my_instance, name, NumpyArrayLike(buffer))
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
                    serial_exception = HDF5_GroupSerializationException(
                        f"We couldn't serialize group named {name} (group is attached in the exception.",
                        e,
                        group=value,
                    )
                    log.exception(serial_exception.msg, serial_exception)
                    raise serial_exception

                # assert get_class_for_name(name) and isinstance(), f"No class found that corresponds to group {name}! Make sure there's a corresponding dataclass named {name} in this module scope!"

                try:
                    this_instance = ThisClass.from_group(value, log=log)
                except AttributeError as e:
                    serial_exception = HDF5_GroupSerializationException(
                        f"We couldn't serialize group named {name!s} from class {ThisClass!s}. It appears {ThisClass!s} doesn't implement the {HDF5_GroupSerializing.__name__} protocol. Group is attached in the exception.",
                        e,
                        group=value,
                    )
                    log.exception(serial_exception.msg, serial_exception)
                    raise serial_exception

                object.__setattr__(my_instance, name, this_instance)

        return my_instance


class HDF5_GroupSerialableDataclass(HDF5_GroupSerializable):
    def as_group(self, parent_group: HDF5_Group, log: Optional[Logger] = None) -> HDF5_Group:
        log = log if log is not None else getLogger()

        """Returns this object as an HDF5 Group."""
        my_group: HDF5_Group = parent_group.require_group(self.name())

        for field_name, field_value in vars(self).items():
            if isinstance(field_value, HDF5_GroupSerializable):
                # This value is actually its own group.
                # So we create a new group rooted at our dataclass's group
                # And assign it the value of whatever the group of the value is.
                # new_group = my_group.require_group(field_name)
                field_value.as_group(my_group)
            elif isinstance(field_value, HDF5_DatasetSerializable):
                field_value.as_a(my_group, log)
            else:
                my_group.create_attr(field_name, field_value)
        return my_group

    @classmethod
    def from_group(
        cls, group: HDF5_Group, log: Optional[Logger] = None
    ) -> HDF5_GroupSerialableDataclass:
        log = log if log is not None else getLogger()
        if not log:
            log = getLogger()
        my_instance = cls.__new__(cls)

        # First, copy over attrs:
        my_instance.create_attrs(group.get_attrs())

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
                    serial_exception = HDF5_GroupSerializationException(
                        f"We couldn't serialize group named {name} (group is attached in the exception.",
                        e,
                        group=value,
                    )
                    log.exception(serial_exception.msg, serial_exception)
                    raise serial_exception

                # assert get_class_for_name(name) and isinstance(), f"No class found that corresponds to group {name}! Make sure there's a corresponding dataclass named {name} in this module scope!"

                try:
                    this_instance = ThisClass.from_group(value, log=log)
                except AttributeError as e:
                    serial_exception = HDF5_GroupSerializationException(
                        f"We couldn't serialize group named {name!s} from class {ThisClass!s}. It appears {ThisClass!s} doesn't implement the {HDF5_GroupSerializing.__name__} protocol. Group is attached in the exception.",
                        e,
                        group=value,
                    )
                    log.exception(serial_exception.msg, serial_exception)
                    raise serial_exception

                object.__setattr__(my_instance, name, this_instance)

        return my_instance


###########################################################
#
#                      HDF5 Datasets
#
###########################################################


class HDF5_DatasetSerializing(HDF5_Dataset, HDF5_Serializing):
    """Objects adhering to the `HDF5_GroupSerializable` can be written to and
    read directly from hd5 Groups.
    """

    def name(self) -> str:
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
        return self.__class__.__name__


class HDF5_DatasetSerializable(HDF5_DatasetSerializing):
    @classmethod
    def from_a(
        cls, a: Union[HDF5_Dataset, HDF5_Group], log: Optional[Logger] = None
    ) -> HDF5_DatasetSerializable:
        # Assume A is the parent group
        # Assuming we're storing a numpy array as this dataset

        # Copies the values into our buffer
        try:
            buffer = np.empty(a.shape, dtype=a.dtype)
            a.read_direct(buffer)
            data = NumpyArrayLike(buffer)

            return HDF5_DatasetSerializable(cls.__new__(cls, buffer))

        except AttributeError as e:
            log.error("Could not convert to HDF5_DatasetSerializable from: {a!r}")
            raise e
        # serialized = cls.__new__(cls)
        return

    def as_a(self, a: HDF5_Group, log: Optional[Logger] = None) -> HDF5_Dataset:
        dataset = HDF5_Dataset(a.require_dataset(self.name(), shape=self.shape, dtype=self.dtype))
        return dataset

    def update(self, log: Optional[Logger] = None):
        self.as_a(self._group.parent, log=log)


class HDF5_DatasetSerialableDataclass(HDF5_DatasetSerializable):
    def as_dataset(self, parent_group: HDF5_Group, log: Optional[Logger] = None) -> HDF5_Dataset:
        log = log if log is not None else getLogger()

        """Returns this object as an HDF5 Group."""
        dataset: HDF5_Dataset = super().as_a(parent_group)

        dataset.create_attrs(vars(self))
        # for field_name, field_value in vars(self).items():
        #     dataset.create_attr(field_name, field_value)
        return dataset

    @classmethod
    def from_dataset(
        cls, dataset: HDF5_Dataset, log: Optional[Logger] = None
    ) -> HDF5_DatasetSerialableDataclass:
        log = log if log is not None else getLogger()
        if not log:
            log = getLogger()
        my_instance = cls.__new__(dataset)

        # First, copy over attrs:
        for name, value in dataset.objects_from_attrs():
            object.__setattr__(my_instance, name, value)

        return my_instance
