"""
===================
fast5.py
===================

Classes for reading, writing and validating fast5 files.
"""
import json
from collections import namedtuple
from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Dict, List, NewType, Optional, Union

import h5py
from poretitioner.utils.filtering import PATH as FILTER_PATH
from poretitioner.utils.filtering import FilterPlugin, RangeFilter

from .application_info import get_application_info
from .logger import Logger, getLogger
from .signals import (
    CaptureMetadata,
    ChannelCalibration,
    FractionalizedSignal,
    RawSignal,
    VoltageSignal,
)
from .utils.configuration import FilterConfig, SegmentConfiguration
from .utils.core import NumpyArrayLike, PathLikeOrString

__all__ = ["BulkFile", "CaptureFile", "channel_path_for_read_id", "signal_path_for_read_id"]

# The name of the root of a HDF file. Also attainable from h5py.File.name: https://docs.h5py.org/en/latest/high/file.html#reference
FAST5_ROOT = "/"


@dataclass(frozen=True)
class KEY:
    """Group and attribute IDs used in Fast5 files.
    """

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


@dataclass(frozen=True)
class CAPTURE_PATH:
    CONTEXT_TAGS = "/Meta/context_tags"
    TRACKING_ID = "/Meta/tracking_id"
    SEGMENTATION = "/Meta/Segmentation"
    CAPTURE_WINDOWS = "/Meta/Segmentation/capture_windows"
    CONTEXT_ID = "/Meta/Segmentation/context_id"
    FILTERS = "/Meta/Segmentation/filters"


class SubRun(namedtuple("SubRun", ["id", "offset", "duration"])):
    """ If the bulk fast5 contains multiple runs (shorter sub-runs throughout
    the data collection process), this can be used to record additional
    context about the sub run: (id : str, offset : int, and
    duration : int). `id` is the identifier for the sub run,
    `offset` is the time the sub run starts in the bulk fast5,
    measured in #/time series points.
    """

    id: str
    offset: int
    duration: int


def format_read_id(read_id: str):
    return read_id if "read" in read_id else f"read_{read_id}"


def channel_path_for_read_id(read_id: str, root: str = FAST5_ROOT) -> PathLikeOrString:
    """Generates an HDFS group path for a read_id's channel.

    Parameters
    ----------
    read_id : str, optional.
        Read ID of the channel read. None by default.

    Returns
    -------
    str
        Correctly formatted channel path.
    """
    read_id = (
        read_id if "read" in read_id else f"read_{read_id}"
    )  # Conditionally adds the "read_" prefix, for backwards compatibility with capture files that didn't use this prefix.
    read_id_path = PurePosixPath(root, read_id)
    channel_path = add_channel_id_to_path(read_id_path)

    if "read" in read_id:
        channel_path = PurePosixPath(root, read_id, KEY.CHANNEL_ID)
    else:
        channel_path = PurePosixPath(root, f"read_{read_id}", KEY.CHANNEL_ID)
    return channel_path


def signal_path_for_read_id(read_id: str, root: str = FAST5_ROOT) -> PathLikeOrString:
    """Generates an HDFS group path for a read_id's signal.

    Parameters
    ----------
    read_id : str
        Read ID of the signal read.

    Returns
    -------
    str
        Correctly formatted signal path.
    """
    if "read" in read_id:
        signal_path = PurePosixPath(root, read_id, KEY.SIGNAL)
    else:
        signal_path = PurePosixPath(root, f"read_{read_id}", KEY.SIGNAL)

    return signal_path


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
    path_with_signal = PurePosixPath(base, KEY.SIGNAL)
    return path_with_signal


def add_channel_id_to_path(base: PathLikeOrString) -> PathLike:
    # We're using PurePosixPath, instead of Path, to guarantee that the path separator will be '/' (i.e. FAST5_ROOT) (instead of using the operating system default)
    path_with_channel_id = PurePosixPath(base, KEY.CHANNEL_ID)
    return path_with_channel_id


class BaseFile:
    def __init__(self, filepath: PathLikeOrString, mode: str = "r", logger: Logger = getLogger()):
        """Base class for interfacing with Fast5 files. This class should not be instantiated directly, instead it should be subclassed.
        Most of the time, nanopore devices/software write data in an HDFS [1] file format called Fast5.
        We expect a certain format for these files, and write our own.

        [1] - https://support.hdfgroup.org/HDF5/whatishdf5.html

        Parameters
        ----------
        filepath : PathLike
            Path to the fast5 file to interface with.

        mode : str, optional
            File mode, valid modes are:
            - "r" 	Readonly, file must exist (default)
            - "r+" 	Read/write, file must exist
            - "w" 	Create file, truncate if exists
            - "w-" or "x" 	Create file, fail if exists
            - "a" 	Read/write if exists, create otherwise

        logger : Logger, optional
            Logger to use, by default getLogger()

        Raises
        ------
        OSError
            File couldn't be opened (e.g. didn't exist, OS 'Resource temporarily unavailable'). Details in message.
        ValueError
            File had validation errors, details in message.
        """

        self.filepath = Path(filepath).expanduser().resolve()
        self.f5 = h5py.File(self.filepath, mode)
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
            rng = attrs.get("range")
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

        calibration = ChannelCalibration(offset=offset, rng=rng, digitisation=digi)
        return calibration


class BulkFile(BaseFile):
    # TODO: Katie Q: Work with Katie to determine how we want to handle validation.

    def __init__(
        self, bulk_filepath: PathLikeOrString, mode: str = "r", logger: Logger = getLogger()
    ):
        """Interfaces with Bulk fast5 files. The Bulk fast5 refers to the file format generated by Oxford Nanopore MinKnow devices.

        Parameters
        ----------
        bulk_filepath : PathLikeOrString
            File path to the bulk fast 5 file.

        mode : str, optional
            File mode, valid modes are:
            - "r" 	Readonly, file must exist (default)
            - "r+" 	Read/write, file must exist
            - "w" 	Create file, truncate if exists
            - "w-" or "x" 	Create file, fail if exists
            - "a" 	Read/write if exists, create otherwise

        logger : Logger, optional
            Logger to use, by default getLogger()

        Raises
        ------
        OSError
            Bulk file couldn't be opened (e.g. didn't exist, OS 'Resource temporarily unavailable'). Details in message.
        ValueError
            Bulk file had validation errors, details in message.
        """
        super().__init__(bulk_filepath, mode=mode, logger=logger)
        if not self.filepath.exists():
            raise OSError(
                f"Bulk fast5 file does not exist at path: {bulk_filepath}. Make sure the bulk file is in this location."
            )
        self.validate(log=logger)

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
            run_id = self.tracking_id["run_id"].decode("utf-8")
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
            Start point.
        end : Optional[int]
            End point.

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
    def context_tags(self) -> Dict:
        context_tags = self.f5.get(BULK_PATH.CONTEXT_TAGS)
        result = {} if context_tags is None else dict(context_tags.attrs)
        return result

    @property
    def tracking_id(self) -> Dict:
        tracking_id = self.f5.get(BULK_PATH.TRACKING_ID)
        result = {} if tracking_id is None else dict(tracking_id.attrs)
        return result


class CaptureFile(BaseFile):
    def __init__(
        self, capture_filepath: PathLikeOrString, mode: str = "r", logger: Logger = getLogger()
    ):
        """Capture file.

        Parameters
        ----------
        capture_filepath : PathLikeOrString
            Path to the capture file. Capture files are the result of running `poretitioner segment` on a bulk file.
        mode : str, optional
            File mode, valid modes are:
            - "r" 	Readonly, file must exist (default)
            - "r+" 	Read/write, file must exist
            - "w" 	Create file, truncate if exists
            - "w-" or "x" 	Create file, fail if exists
            - "a" 	Read/write if exists, create otherwise
        logger : Logger, optional
            Logger to use, by default getLogger()

        Raises
        ------
        OSError
            Capture file couldn't be opened (e.g. didn't exist, OS 'Resource temporarily unavailable'). Details in message.
        ValueError
            Capture file had validation errors, details in message.
        """
        logger.debug(f"Creating capture file at {capture_filepath} in mode ({mode})")
        super().__init__(capture_filepath, mode=mode, logger=logger)
        if not self.filepath.exists():
            error_msg = f"Capture fast5 file does not exist at path: {self.filepath}. Make sure the capture file is in this location."
            raise OSError(error_msg)

    def initialize_from_bulk(
        self,
        bulk_f5: BulkFile,
        filters: List[RangeFilter],
        segment_config: SegmentConfiguration,
        sub_run: Optional[SubRun] = None,
    ):
        # Only supporting range filters for now (MVP). This should be expanded to all FilterPlugins: https://github.com/uwmisl/poretitioner/issues/67

        # Referencing spec v0.1.1

        # /Meta/context_tags
        capture_context_tags_group = self.f5.create_group(CAPTURE_PATH.CONTEXT_TAGS)
        for key, value in bulk_f5.context_tags.items():
            capture_context_tags_group.attrs.create(key, value)

        bulk_f5_fname = bulk_f5.filename

        filename_length = len(bulk_f5_fname)
        dtype = f"S{filename_length}"  # Need to save the length of the string.
        capture_context_tags_group.attrs.create("bulk_filename", bulk_f5_fname, dtype=dtype)

        sampling_frequency = bulk_f5.sampling_rate
        capture_context_tags_group.attrs.create("sample_frequency", sampling_frequency)

        # /Meta/tracking_id
        capture_tracking_id_group = self.f5.create_group(CAPTURE_PATH.TRACKING_ID)
        for key, value in bulk_f5.tracking_id.items():
            capture_tracking_id_group.attrs.create(key, value)

        if sub_run is not None:
            id = sub_run.id
            offset = sub_run.offset
            duration = sub_run.duration
            if id is not None:
                capture_tracking_id_group.attrs.create("sub_run_id", id, dtype=f"S{len(id)}")
            if offset is not None:
                capture_tracking_id_group.attrs.create("sub_run_offset", offset)
            if duration is not None:
                capture_tracking_id_group.attrs.create("sub_run_duration", duration)

        # /Meta/Segmentation
        # TODO: define config param structure : https://github.com/uwmisl/poretitioner/issues/27
        # config = {"param": "value",
        #           "filters": {"f1": (min, max), "f2: (min, max)"}}
        capture_segmentation_group = self.f5.create_group(CAPTURE_PATH.SEGMENTATION)
        # print(__name__)
        version = get_application_info().version
        segmenter_name = __name__
        capture_segmentation_group.attrs.create(
            "segmenter", __name__, dtype=f"S{len(segmenter_name)}"
        )
        capture_segmentation_group.attrs.create(
            "segmenter_version", version, dtype=f"S{len(version)}"
        )
        filter_group = self.f5.create_group(CAPTURE_PATH.FILTERS)
        context_id_group = self.f5.create_group(CAPTURE_PATH.CONTEXT_ID)
        capture_windows_group = self.f5.create_group(CAPTURE_PATH.CAPTURE_WINDOWS)

        # KATIE Q: Is it same to assume that whatever filter configuration is defined in the
        # filter section will be used in subsequent steps?
        # Only supporting range filters for now (MVP): https://github.com/uwmisl/poretitioner/issues/67
        if segment_config is None:
            raise ValueError("No segment configuration provided.")
        else:
            for key, value in vars(segment_config).items():
                save_value = json.dumps(value)
                context_id_group.create_dataset(key, data=save_value)

        for filter_plugin in filters:
            name = filter_plugin.name()

            # `isinstance` is an anti-pattern, pls don't use in production.
            if isinstance(filter_plugin, RangeFilter):
                maximum = filter_plugin.maximum
                minimum = filter_plugin.minimum

                self.log.debug(f"Filtering for {name}: ({minimum}, {maximum})")
                if minimum is None:
                    minimum = -1
                if maximum is None:
                    maximum = -1
                if minimum == -1 and maximum == -1:
                    continue
                filter_group.create_dataset(name, data=(minimum, maximum))
            # Based on the example code, it doesn't seem like we write anything for ejected filter?

    def validate(self, capture_filepath: PathLikeOrString, log: Logger = getLogger()):
        """Make sure this represents a valid capture/segmented poretitioner file.

        Raises
        ------
        ValueError
            Capture/segmented file had some validation issues, details in message.
        """
        pass

    @property
    def sampling_rate(self) -> int:
        """Retrieve the sampling rate from a capture fast5 file. Units: Hz.

        Also referred to as the sample rate, sample frequency, or sampling
        frequency. Or some say Kosm.

        Returns
        -------
        int
            Sampling rate
        """

        context_tag_path = "/Meta/context_tags"
        sample_frequency_key = "sample_frequency"
        sample_rate_path = "/Meta"
        sample_rate_key = "sample_rate"

        sample_frequency = None
        try:
            sample_frequency = int(self.f5[context_tag_path].attrs[sample_frequency_key])
        except KeyError:
            # Try checking the Meta group as a fallback.
            try:
                sample_rate = int(self.f5[sample_rate_path].attrs[sample_rate_key])
                sample_frequency = sample_rate
            except KeyError:
                error_msg = f"Sampling rate not present in bulk file '{self.f5.filename}'. Make sure a sampling frequency is specified either at '{sample_rate_path}' with attribute '{sample_frequency_key}', or as a fallback, '{sample_rate_path}' with attribute '{sample_rate_key}'"
                self.log.error(error_msg)
                raise ValueError(error_msg)

        rate = sample_frequency
        return rate

    @property
    def reads(self, root: str = FAST5_ROOT) -> List[str]:
        root_group = self.f5.get(root)
        potential_reads = [] if not root_group else root_group.keys()
        reads = [read for read in potential_reads if self.is_read(read)]
        return reads

    @property
    def filtered_reads(self):
        # TODO: Implement filtering here  to only return reads that pass a filter. https://github.com/uwmisl/poretitioner/issues/67
        return self.reads

    def is_read(self, path: PathLikeOrString) -> bool:
        """Whether this path points to a capture read.

        Parameters
        ----------
        path : PathLikeOrString
            Path that may point to a read.

        Returns
        -------
        bool
            True if and only if the path represents a read.
        """
        # Reads should have Signal and channelId groups.
        # If a path has both of theses, we consider the group a signal
        channel_path = str(add_channel_id_to_path(path))
        signal_path = str(add_signal_to_path(path))
        return self.f5.get(channel_path) and self.f5.get(signal_path)

    def get_fractionalized_read(
        self, read_id: str, start: Optional[int] = None, end: Optional[int] = None
    ) -> FractionalizedSignal:
        """Gets the fractionalized signal from this read.

        Parameters
        ----------
        read_id : str
            Read to get the fractionalized signal from.
        start : Optional[int], optional
            Where to start the read, by default None
        end : Optional[int], optional
            Where to end the read, by default None

        Returns
        -------
        FractionalizedSignal
            Fractionalized signal from times `start` to `end`.
        """
        signal_path = signal_path_for_read_id(read_id)
        channel_path = channel_path_for_read_id(read_id)
        open_channel_pA = self.f5[signal_path].attrs[KEY.OPEN_CHANNEL_PA]
        calibration = self.get_channel_calibration_for_read(read_id)
        raw_signal: RawSignal = self.f5.get(signal_path)[start:end]
        channel_number = self.f5.get(channel_path).attrs["channel_number"]
        fractionalized = FractionalizedSignal(
            raw_signal,
            channel_number,
            calibration,
            open_channel_pA,
            read_id=read_id,
            do_conversion=True,
        )

        return fractionalized

    def get_channel_calibration_for_read(self, read_id: str) -> ChannelCalibration:
        """Retrieve the channel calibration for a specific read in a segmented fast5 file (i.e. CaptureFile).
        This is used for properly scaling values when converting raw signal to actual units.

        Note: using UK spelling of digitization for consistency w/ file format

        Parameters
        ----------
        read_id : str
            Read id to retrieve raw signal. Can be formatted as a path ("read_xxx...")
            or just the read id ("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").

        Returns
        -------
        ChannelCalibration
            Channel calibration Offset, range, and digitisation values.
        """
        channel_path = channel_path_for_read_id(read_id)
        calibration = self.get_channel_calibration_for_path(channel_path)
        return calibration

    def get_capture_metadata_for_read(self, read_id: str) -> CaptureMetadata:
        """Retrieve the capture metadata for given read.

        Parameters
        ----------
        read_id : str
            Which read to fetch the metadata for.

        Returns
        -------
        CaptureMetadata
            Metadata around the captures in this read.
        """
        channel_path = channel_path_for_read_id(read_id)
        signal_path = signal_path_for_read_id(read_id)

        channel_number = self.f5[channel_path].attrs["channel_number"]
        sampling_rate = self.f5[channel_path].attrs["sampling_rate"]
        calibration = self.get_channel_calibration_for_read(read_id)

        start_time_bulk = self.f5[signal_path].attrs["start_time_bulk"]
        start_time_local = self.f5[signal_path].attrs["start_time_local"]
        duration = self.f5[signal_path].attrs["duration"]
        ejected = self.f5[signal_path].attrs["ejected"]
        voltage_threshold = self.f5[signal_path].attrs["voltage"]
        open_channel_pA = self.f5[signal_path].attrs[KEY.OPEN_CHANNEL_PA]

        cap = CaptureMetadata(
            read_id,
            start_time_bulk,
            start_time_local,
            duration,
            ejected,
            voltage_threshold,
            open_channel_pA,
            channel_number,
            calibration,
            sampling_rate,
        )
        return cap

    def write_capture(self, raw_signal: RawSignal, metadata: CaptureMetadata):
        """Write a single capture to the specified capture fast5 file (which has
        already been created via create_capture_fast5()).

        Parameters
        ----------
        raw_signal : RawSignal
            Time series of nanopore current values (in units of pA).
        metadata: CaptureMetadata
            Details about this capture.
        """
        read_id = metadata.read_id
        f5 = self.f5
        signal_path = str(signal_path_for_read_id(read_id))
        f5[signal_path] = raw_signal
        f5[signal_path].attrs["read_id"] = read_id
        f5[signal_path].attrs["start_time_bulk"] = metadata.start_time_bulk
        f5[signal_path].attrs["start_time_local"] = metadata.start_time_local
        f5[signal_path].attrs["duration"] = metadata.duration
        f5[signal_path].attrs["ejected"] = metadata.ejected
        f5[signal_path].attrs["voltage"] = metadata.voltage_threshold
        f5[signal_path].attrs[KEY.OPEN_CHANNEL_PA] = metadata.open_channel_pA

        channel_path = str(channel_path_for_read_id(read_id))
        f5.create_group(channel_path)
        f5[channel_path].attrs["channel_number"] = metadata.channel_number
        f5[channel_path].attrs["digitisation"] = metadata.calibration.digitisation
        f5[channel_path].attrs["range"] = metadata.calibration.rng
        f5[channel_path].attrs["offset"] = metadata.calibration.offset
        f5[channel_path].attrs["sampling_rate"] = metadata.sampling_rate
        f5[channel_path].attrs[KEY.OPEN_CHANNEL_PA] = metadata.open_channel_pA

    def write_filter_results(self, read_ids: List[str]):
        # For all read_ids that passed the filter (AKA reads that were passed in),
        # create a hard link in the filter_path to the actual read's location in
        # the fast5 file.
        set_matching_read_ids = set(self.reads)
        passing_read_ids = [read_id for read_id in read_ids if read_id in set_matching_read_ids]
        for read_id in passing_read_ids:
            read_path = format_read_id(read_id)
            read_grp = self.f5.get(read_path)
            self.log.debug(read_grp)
            filter_read_path = FILTER_PATH.filter_pass_path_for_read_id(read_id)
            # Create a hard link from the filter read path to the actual read path
            self.f5[filter_read_path] = read_grp
