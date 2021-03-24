from typing import *

from .plugin import Plugin

from ..hdf5.exceptions import HDF5_SerializationException, HDF5_GroupSerializationException, HDF5_DatasetSerializationException


class PluginNotFoundException(AttributeError):
    def __init__(self, msg: str, *args: object, plugin: Optional[Any] = None) -> None:
        super().__init__(msg, *args)
        self.plugin = plugin


class CaptureSchemaVersionException(HDF5_SerializationException):
    """We couldn't serialize this object because it was store with an incompatible schema version."""

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(msg, *args)
