import h5py

from typing import Optional

class HDF5_SerializationException(AttributeError):
    """We couldn't serialize a data class to/from a HDF5 file."""

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.msg = msg

class HDF5_GroupSerializationException(HDF5_SerializationException):
    """We couldn't serialize a group class to/from a HDF5 file."""

    def __init__(
        self, msg: str, *args: object, group: Optional[h5py.Group] = None
    ) -> None:
        super().__init__(msg, *args)
        self.group = group

class HDF5_DatasetSerializationException(HDF5_SerializationException):
    """We couldn't serialize a dataset to/from a HDF5 file."""

    def __init__(
        self, msg: str, *args: object, dataset: Optional[h5py.Dataset] = None
    ) -> None:
        super().__init__(msg, *args)
        self.dataset = dataset
