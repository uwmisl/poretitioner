from typing import *
import h5py
from .plugin import Plugin

class HDF5SerializationException(AttributeError):
    """We couldn't serialize a data class from a HDF5 file."""

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.msg = msg

class HDF5GroupSerializationException(HDF5SerializationException):
    """We couldn't serialize a data class from a HDF5 file."""

    def __init__(
        self, msg: str, *args: object, group: Optional[h5py.Group] = None
    ) -> None:
        super().__init__(msg, *args)
        self.group = group

class PluginNotFoundException(AttributeError):
    def __init__(
        self, msg: str, *args: object, plugin: Optional[Any] = None
    ) -> None:
        super().__init__(msg, *args)
        self.plugin = plugin

class CaptureSchemaVersionException(HDF5SerializationException):
    """We couldn't serialize this object because it was store with an incompatible schema version."""

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(msg, *args)
