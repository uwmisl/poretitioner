"""
===================
fast5.py
===================

Classes for reading, writing and validating fast5 files.
"""
from __future__ import annotations

import dataclasses
import json
from abc import abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PosixPath, PurePosixPath
from typing import *  # I know people don't like import *, but I think it has benefits for types (doesn't impede people from being generous with typing)

import h5py

from .application_info import get_application_info
from .hdf5.hdf5 import (
    Fast5File,
    HasFast5,
    HDF5_DatasetSerialableDataclass,
    HDF5_Group,
    HDF5_GroupSerialableDataclass,
    HDF5_GroupSerializable,
    HDF5_GroupSerializing,
    hdf5_dtype,
)
from .logger import Logger, getLogger
from .signals import CaptureMetadata, FractionalizedSignal, RawSignal, VoltageSignal
from .utils.classify import CLASSIFICATION_PATH
from .utils.configuration import SegmentConfiguration
from .utils.core import NumpyArrayLike, PathLikeOrString, ReadId, dataclass_fieldnames
from .utils.filtering import FilterSet

__all__ = [
    "BulkFile",
    "CaptureFile",
    "channel_path_for_read_id",
    "signal_path_for_read_id",
    "ContextTagsBase",
    "ContextTagsBulk",
]

# The name of the root of a HDF file. Also attainable from h5py.File.name: https://docs.h5py.org/en/latest/high/file.html#reference
FAST5_ROOT = "/"


@dataclass(frozen=True)
class ContextTagsBase(HDF5_GroupSerialableDataclass):
    """A Context Tags group in a Fast5 file.

    We have separate classes for Bulk files `ContextTagsBulk`,
    and Capture files `ContextTagsCapture` because their fields
    vary slightly.
    """

    experiment_type: str
    sample_frequency: str

    department: str
    local_bc_comp_model: str
    local_bc_temp_model: str
    user_filename_input: str

    @classmethod
    def name(cls):
        # We have to override the default, because the context tags
        # are stored under a different name than the class name
        return "context_tags"


@dataclass(frozen=True)
class ContextTagsBulk(ContextTagsBase):
    """A Context Tags group in a Bulk Fast5 file.

    We have separate classes for Bulk files `ContextTagsBulk`,
    and Capture files `ContextTagsCapture` because their fields
    vary slightly.
    """

    filename: str


@dataclass(frozen=True)
class TrackingId(HDF5_GroupSerialableDataclass):
    # Q: Why are we violating PEP8 conventions in having a lower-cased class name?
    # A: To match how ONT stores the group in the fast5 file.
    #    This lets us serialize/deserialize by class name directly.
    asic_id: str
    asic_id_eeprom: str
    asic_temp: str
    auto_update: str
    auto_update_source: str
    bream_core_version: str
    bream_is_standard: str
    bream_ont_version: str
    device_id: str
    exp_script_name: str
    exp_script_purpose: str
    exp_start_time: str
    flow_cell_id: str
    heatsink_temp: str
    hostname: str
    installation_type: str
    local_firmware_file: str
    operating_system: str
    protocol_run_id: str
    protocols_version: str
    run_id: str
    sample_id: str
    usb_config: str
    version: str

    @classmethod
    def name(cls):
        # We have to override the default, because the context tags
        # are stored under a different name than the class name
        return "tracking_id"


@dataclass(frozen=True)
class KEY:
    """Group and attribute IDs used in Fast5 files."""

    # Key name for where channel info is stored in Fast5 files.
    CHANNEL_ID = "channel_id"
    # Key name for where Signal info is stored in Fast5 files.
    SIGNAL = "Signal"

    META = "Meta"
    SEGMENTATION = "Segmentation"
    OPEN_CHANNEL_PA = "open_channel_pA"


@dataclass(frozen=True)
class BULK_PATH:
    UNIQUE_GLOBAL_KEY = "/UniqueGlobalKey"
    CONTEXT_TAGS = "/UniqueGlobalKey/context_tags"
    TRACKING_ID = "/UniqueGlobalKey/tracking_id"


def format_read_id(read_id: str) -> ReadId:
    """Take a read_id and ensure it's prefixed with "read_".

    Parameters
    ----------
    read_id : ReadId
        Read_id, which may or may not be prefixed with "read_".

    Returns
    -------
    ReadId
        Read_id string, which uniquely identifies a read.
    """
    return ReadId(read_id) if read_id.startswith("read_") else ReadId(f"read_{read_id}")


def channel_path_for_read_id(read_id: ReadId, root: str = FAST5_ROOT) -> PathLikeOrString:
    """Generates an HDFS group path for a read_id's channel.

    Parameters
    ----------
    read_id : ReadId, optional.
        Read ID of the channel read. None by default.

    Returns
    -------
    str
        Correctly formatted channel path.
    """
    read_id = format_read_id(read_id)
    read_id_path = PurePosixPath(root, read_id)
    channel_path = add_channel_id_to_path(read_id_path)
    channel_path = PurePosixPath(root, read_id, KEY.CHANNEL_ID)

    return str(channel_path)


def signal_path_for_read_id(read_id: ReadId, root: str = FAST5_ROOT) -> PathLikeOrString:
    """Generates an HDFS group path for a read_id's signal.

    Parameters
    ----------
    read_id : ReadId
        Read ID of the signal read.

    Returns
    -------
    str
        Correctly formatted signal path.
    """
    read_id = format_read_id(read_id)
    signal_path = PurePosixPath(root, read_id, KEY.SIGNAL)

    return str(signal_path)


def add_signal_to_path(base: PathLikeOrString) -> PathLike:
    """Adds the Signal Group key to a path.

    Parameters
    ----------
    base : str | PathLike
        Path to add the Signal path to.

    Returns
    -------
    PathLike
        A HDF5-friendly path that contains the signal group.
    """
    # We're using PurePosixPath, instead of Path, to guarantee that the path separator will be '/' (i.e. FAST5_ROOT) (instead of using the operating system default)
    path_with_signal = str(PurePosixPath(base, KEY.SIGNAL))
    return path_with_signal


def add_channel_id_to_path(base: PathLikeOrString) -> PathLike:
    # We're using PurePosixPath, instead of Path, to guarantee that the path separator will be '/' (i.e. FAST5_ROOT) (instead of using the operating system default)
    path_with_channel_id = str(PurePosixPath(base, KEY.CHANNEL_ID))
    return path_with_channel_id


@dataclass(frozen=True)
class ChannelCalibration(HDF5_GroupSerialableDataclass):
    """
    On the Oxford Nanopore devices, there's an analog to digital converter that converts the raw, unitless analog signal
    transforms the raw current. To convert these raw signals to a measurement of current in picoAmperes,
    we need to know certain values for the channel configuration.


    ⎜   Physics    ⎜  Device  ⎜            Software          ⎜

    ⎜‒‒‒‒‒‒‒‒‒‒‒‒‒⎜‒‒‒‒‒‒‒‒‒⎜‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒⎜

    ⎜             ⎜         ⎜                            ⎜

    ⎜   Current   ⟶  ADC    ⟶    Calibration ⟶  Signal     ⎜

    ⎜  (Amperes)   ⎜ (Unitless) ⎜      (Unitless)    (PicoAmperes)⎜

    ⎜             ⎜         ⎜                            ⎜

    ⎜‒‒‒‒‒‒‒‒‒‒‒‒⎜‒‒‒‒‒‒‒‒‒⎜‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒‒⎜


    Fields
    ----------
    offset : int
        Adjustment to the ADC to each 0pA.
        Essentially, this is the average value the device ADC gives when there's zero current.
        We subtract this from the raw ADC value to account for this, like zero'ing out a scale before weighing.

    range: float
        The range of digitized picoAmperes that can be produced after the analog-to-digital conversion. The ratio of range:digitisation represents the change in picoamperes of current per ADC change of 1.

    digitisation: int
        This is the number of values that can even be produced the Analog-to-Digital. This exists because only a finite number of values can be represnted by a digitized signal.
    """

    offset: float
    range: float
    digitisation: float


class BaseFile(HasFast5):
    def __init__(self, filepath: PathLikeOrString, mode: str = "r", logger: Logger = getLogger()):
        # TODO: Black doesn't like the formatting here, can't figure out why.
        # fmt: off
        """Base class for interfacing with Fast5 files.

        This class should not be instantiated directly, instead it should be subclassed.

        Most of the time, nanopore devices/software write data in an HDFS [1] file format called Fast5.
        We expect a certain format for these files, and write our own.

        [1] - https://support.hdfgroup.org/HDF5/whatishdf5.html

        Parameters
        ----------
        filepath : PathLike
            Path to the fast5 file to interface with.

        mode : str, optional

            File mode, valid modes are:

            "r" 	Readonly, file must exist (default)

            "r+" 	Read/write, file must exist

            "w" 	Create file, truncate if exists

            "w-" or "x" 	Create file, fail if exists

            "a" 	Read/write if exists, create otherwise

        logger : Logger, optional
            Logger to use, by default getLogger()

        Raises
        ------
        OSError
            File couldn't be opened (e.g. didn't exist, OS Resource temporarily unavailable). Details in message.
        ValueError
            File had validation errors, details in message.
        """
        # TODO: Black doesn't like the formatting here, can't figure out why.
        # fmt: on

        self.filepath = Path(filepath).expanduser().resolve()
        self.f5 = HDF5_Group(h5py.File(self.filepath, mode=mode))
        self.filename = self.f5.filename
        self.log = logger
        self.ROOT = self.f5.name  # '/' by default

    def __enter__(self):
        self.f5.__enter__()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.f5.__exit__(exception_type, exception_value, traceback)

    def get_channel_calibration_for_path(self, path: PathLikeOrString) -> ChannelCalibration:
        """Gets the channel calibration

        Parameters
        ----------
        path : PathLikeOrString
            Group path where the channel calibration data is stored.

        Returns
        -------
        ChannelCalibration
            Channel calibration associated with this file.

        Raises
        ------
        ValueError
            Channel calibration is not present at that path, or is malformed.
        """

        offset_key = "offset"
        range_key = "range"
        digitisation_key = "digitisation"

        try:
            attrs = self.f5[path].attrs
        except KeyError:
            error_msg = f"Calibration data does not exist at path '{path}'. Double-check that the channel calibration data is in this location."
            self.log.error(error_msg)
            raise ValueError(error_msg)

        try:
            offset = attrs.get(offset_key)
        except KeyError:
            error_msg = f"Attribute '{offset_key}' does not exist at path '{path}'. Double-check that the {offset_key} attribute exists in this location."
            self.log.error(error_msg)
            raise ValueError(error_msg)

        try:
            range = attrs.get("range")
        except KeyError:
            error_msg = f"Attribute '{range_key}' does not exist at path '{path}'. Double-check that the {range_key} attribute exists in this location."
            self.log.error(error_msg)
            raise ValueError(error_msg)

        try:
            digi = attrs.get("digitisation")
        except KeyError:
            error_msg = f"Attribute '{digitisation_key}' does not exist at path '{path}'. Double-check that the {digitisation_key} attribute exists in this location."
            self.log.error(error_msg)
            raise ValueError(error_msg)

        calibration = ChannelCalibration(offset=offset, range=range, digitisation=digi)
        return calibration


class BulkFile(BaseFile):
    def __init__(
        self, bulk_filepath: PathLikeOrString, mode: str = "r", logger: Logger = getLogger()
    ):
        # TODO: Black doesn't like the formatting here, can't figure out why.
        # fmt: off
        """Interfaces with Bulk fast5 files. The Bulk fast5 refers to the file format generated by Oxford Nanopore MinKnow devices.

        Parameters
        ----------
        bulk_filepath : PathLikeOrString
            File path to the bulk fast 5 file.

        mode : str, optional

            File mode, valid modes are:

            "r" 	Readonly, file must exist (default)

            "r+" 	Read/write, file must exist

            "w" 	Create file, truncate if exists

            "w-" or "x" 	Create file, fail if exists

            "a" 	Read/write if exists, create otherwise

        logger : Logger, optional
            Logger to use, by default getLogger()

        Raises
        ------
        OSError
            Bulk file couldn't be opened e.g. didn't exist,
            OS Resource temporarily unavailable. Details in message.
        ValueError
            Bulk file had validation errors, details in message.
        """
        # TODO: Black doesn't like the formatting here, can't figure out why.
        # fmt: on
        super().__init__(bulk_filepath, mode=mode, logger=logger)
        if not self.filepath.exists():
            raise OSError(
                f"Bulk fast5 file does not exist at path: {bulk_filepath}. Make sure the bulk file is in this location."
            )
        #  self.validate(log=logger)

    def validate(self, log: Logger = getLogger()):
        """Make sure this represents a valid bulk poretitioner file.

        Raises
        ------
        ValueError
            Bulk file had some validation issues, details in message.
        """

        # Accessing these fields with raise an exception if any of them are invalid or not present.

        self.run_id
        self.sampling_rate

        # Should have at least one channel
        channel_number = 1
        self.get_channel_calibration(channel_number)

    @property
    def run_id(self) -> str:
        """Unique identifier for the run that generated this bulk file.

        Returns
        -------
        str
            Unique identifier for this run.

        Raises
        ------
        ValueError
            RunID attribute doesn't exist in the expected group.
        """
        path = "/UniqueGlobalKey/tracking_id"

        validation_errors = []
        try:
            tracking_id = self.f5[path]
        except KeyError:
            error_message = f"Group 'tracking_id' missing from path:{path} in bulk file: {self.f5.filename}. Double check that the bulk file includes this field at that path."
            validation_errors.append(error_message)
            self.log.error(error_message)
            raise ValueError(error_message)
        try:
            run_id = self.f5.get(path).attrs["run_id"].decode("utf-8")
        except KeyError:
            error_message = f"Attribute 'run_id' missing from path:{path} in bulk file '{self.f5.filename}'. Double check that the bulk file includes this field at that path."
            validation_errors.append(error_message)
            self.log.error(error_message)
            raise ValueError(error_message)
        return run_id

    @property
    def sampling_rate(self) -> int:
        """Retrieve the sampling rate from a bulk fast5 file. Units: Hz.

        Also referred to as the sample rate, sample frequency, or sampling
        frequency.

        Returns
        -------
        int
            Sampling rate
        """

        sample_frequency_path = "/UniqueGlobalKey/context_tags"
        sample_frequency_key = "sample_frequency"
        sample_rate_path = "/Meta"
        sample_rate_key = "sample_rate"

        sample_frequency = None
        try:
            sample_frequency = int(self.f5[sample_frequency_path].attrs[sample_frequency_key])
        except KeyError:
            pass  # Try checking the Meta group as a fallback.

        try:
            sample_rate = int(self.f5[sample_rate_path].attrs[sample_rate_key])
        except KeyError:
            error_msg = f"Sampling rate not present in bulk file '{self.f5.filename}'. Make sure a sampling frequency is specified either at '${sample_frequency_path}' with attribute '{sample_frequency_key}', or as a fallback, '{sample_rate_path}' with attribute '{sample_rate_key}'"
            self.log.error(error_msg)
            raise ValueError(error_msg)

        rate = sample_frequency if sample_frequency else sample_rate
        return rate

    def get_channel_calibration(self, channel_number: int) -> ChannelCalibration:
        """Retrieve channel calibration for bulk fast5 file.

        Note: using UK spelling of digitization for consistency w/ file format.

        Parameters
        ----------
        channel_number : int
            Channel number for which to retrieve raw signal.

        Returns
        -------
        ChannelCalibration
            Offset, range, and digitisation values.

        Raises
        ------
        ValueError
            Channel calibration is not present at that path, or is malformed.
        """
        meta_path = f"/Raw/Channel_{channel_number}/Meta"
        try:
            self.f5[meta_path]
        except KeyError:
            error_msg = f"Channel does not exist at path '{meta_path}'. Double-check that the device has channel #{channel_number}. Oxford Nanopore channels are 1-indexed, i.e. the first channel is 1."
            self.log.error(error_msg)
            raise ValueError(error_msg)
        calibration = self.get_channel_calibration_for_path(meta_path)
        return calibration

    def get_raw_signal(
        self, channel_number: int, start: Optional[int] = None, end: Optional[int] = None
    ) -> RawSignal:
        """Retrieve raw signal from open fast5 file.

        Optionally, specify the start and end time points in the time series. If no
        values are given for start and/or end, the default is to include data
        starting at the beginning and/or end of the array.

        The signal is not scaled to units of pA; original sampled values are
        returned.

        Parameters
        ----------
        channel_number : int
            Channel number for which to retrieve raw signal.
        start : Optional[int], optional
            Retrieve a slice of current starting at this time point, by default None
            None will retrieve data starting from the beginning of available data.
        end : Optional[int], optional
            Retrieve a slice of current starting at this time point, by default None
            None will retrieve all data at the end of the array.

        Returns
        -------
        Numpy array
            Array representing sampled nanopore current.
        """
        signal_path = f"/Raw/Channel_{channel_number}/Signal"
        calibration = self.get_channel_calibration(channel_number)

        if start is not None or end is not None:
            signal = self.f5.get(signal_path)[start:end]
        else:
            signal = self.f5.get(signal_path)[()]

        raw = RawSignal(signal, channel_number, calibration)
        return raw

    def get_voltage(self, start=None, end=None) -> VoltageSignal:
        """Retrieve the bias voltage.

        Parameters
        ----------
        start : Optional[int]
            Start point (in observations).
        end : Optional[int]
            End point (in observations).

        Returns
        -------
        VoltageSignal[int]
            Voltage(s) in millivolts (mV).
        """
        bias_voltage_multiplier = 5  # Signal changes in increments of 5 mV.
        metadata = self.f5.get("/Device/MetaData")
        voltages = VoltageSignal(metadata["bias_voltage"][start:end] * bias_voltage_multiplier)
        return voltages

    @property
    def context_tags_group(self) -> HDF5_Group:
        return HDF5_Group(self.f5[BULK_PATH.CONTEXT_TAGS])

    @property
    def tracking_id_group(self) -> HDF5_Group:
        return HDF5_Group(self.f5[BULK_PATH.TRACKING_ID])
