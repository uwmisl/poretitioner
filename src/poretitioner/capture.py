from __future__ import annotations

from dataclasses import dataclass
from logging import Logger, getLogger
from pathlib import PosixPath
from typing import Optional, Union

import h5py
import numpy as np

from .application_info import get_application_info
from .fast5s import (
    FAST5_ROOT,
    KEY,
    BaseFile,
    BulkFile,
    ChannelCalibration,
    ContextTagsBase,
    ContextTagsBulk,
    RawSignal,
    TrackingId,
)
from .hdf5 import (
    HasAttrs,
    HDF5_Attributes,
    HDF5_Dataset,
    HDF5_DatasetSerialableDataclass,
    HDF5_DatasetSerializable,
    HDF5_DatasetSerializationException,
    HDF5_Group,
    HDF5_GroupSerialableDataclass,
    HDF5_GroupSerializable,
    HDF5_GroupSerializationException,
    HDF5_GroupSerializing,
    HDF5_SerializationException,
    HDF5_Type,
    NumpyArrayLike,
    get_class_for_name,
    hdf5_dtype,
)
from .signals import HDF5_Signal
from .utils.classify import CLASSIFICATION_PATH
from .utils.configuration import SegmentConfiguration
from .utils.core import PathLikeOrString, ReadId, Window, WindowsByChannel, dataclass_fieldnames
from .utils.filtering import (
    FILTER_PATH,
    FilterPlugin,
    Filters,
    FilterSet,
    HDF5_FilterSet,
    RangeFilter,
)


@dataclass(frozen=True)
class CAPTURE_PATH:
    ROOT = "/"
    CONTEXT_TAGS = "/Meta/context_tags"
    TRACKING_ID = "/Meta/tracking_id"
    SUB_RUN = "/Meta/tracking_id/sub_run"
    SEGMENTATION = "/Meta/Segmentation"
    CAPTURE_WINDOWS = "/Meta/Segmentation/capture_windows"
    CONTEXT_ID = "/Meta/Segmentation/context_id"
    CAPTURE_CRITERIA = "/Meta/Segmentation/capture_criteria"

    @classmethod
    def FOR_READ_ID(cls, read_id: ReadId) -> str:
        path = str(PosixPath(CAPTURE_PATH.ROOT, read_id))
        return path

    @classmethod
    def CHANNEL_FOR_READ_ID(cls, read_id: ReadId) -> str:
        path = str(PosixPath(CAPTURE_PATH.FOR_READ_ID(read_id), KEY.CHANNEL_ID))
        return path

    @classmethod
    def SIGNAL_FOR_READ_ID(cls, read_id: ReadId) -> str:
        path = str(PosixPath(CAPTURE_PATH.FOR_READ_ID(read_id), KEY.SIGNAL))
        return path


@dataclass(frozen=True)
class ContextTagsCapture(ContextTagsBase):
    """A Context Tags group in a Capture Fast5 file.

    We have separate classes for Bulk files `ContextTagsBulk`,
    and Capture files `ContextTagsCapture` because their fields
    vary slightly.
    """

    bulk_filename: str
    filename: str

    @classmethod
    def from_context_tags_bulk_group(
        cls, capture_filename: str, bulk_tags_group: HDF5_Group
    ) -> ContextTagsCapture:
        capture_fields = dataclass_fieldnames(cls)
        context_tags = {"filename": capture_filename}
        for key, value in bulk_tags_group.objects_from_attrs().items():
            if key == "filename":
                # "Filename" in Bulk is named "bulk_filename" in Capture.
                context_tags["bulk_filename"] = value
            else:
                context_tags[key] = value

        tags = cls.__new__(cls)
        tags.__init__(**context_tags)
        return tags


@dataclass(frozen=True)
class Channel(HDF5_GroupSerialableDataclass):
    """Channel-specific information saved for each read."""

    channel_number: int
    calibration: ChannelCalibration
    open_channel_pA: int
    sampling_rate: int

    def name(self) -> str:
        channel_id = f"channel_id"
        return channel_id

    def add_to_group(self, parent_group: HDF5_Group, log: Optional[Logger] = None) -> HDF5_Group:
        log = log if log is not None else getLogger()

        """Returns this object as an HDF5 Group."""
        my_group: HDF5_Group = super().add_to_group(parent_group)

        for field_name, field_value in vars(self).items():
            if isinstance(field_value, HDF5_GroupSerializable):
                # This value is actually its own group.
                # So we create a new group rooted at our dataclass's group
                # And assign it the value of whatever the group of the value is.
                my_group.require_group(field_name)
                field_group = field_value.add_to_group(my_group, log=log)
            elif isinstance(field_value, ChannelCalibration):
                for calibration_key, value in vars(field_value).items():
                    my_group.create_attr(calibration_key, value)
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


@dataclass(frozen=True)
class SubRun(HDF5_GroupSerialableDataclass):
    """If the bulk fast5 contains multiple runs (shorter sub-runs throughout
    the data collection process), this can be used to record additional
    context about the sub run: (id : str, offset : int, and
    duration : int). `id` is the identifier for the sub run,
    `offset` is the time the sub run starts in the bulk fast5,
    measured in #/time series points.
    """

    id: str
    offset: int
    duration: int


@dataclass(frozen=True)
class CaptureTrackingId(TrackingId):
    sub_run: SubRun


@dataclass(frozen=True, init=False)
class CaptureWindows(WindowsByChannel):
    def name(self) -> str:
        return "capture_windows"


@dataclass(init=False)
class Read(HDF5_GroupSerialableDataclass):
    channel_id: Channel
    Signal: HDF5_Signal  # This is the raw signal, fresh from the Fast5.

    def __init__(
        self, parent_group: HDF5_Group, read_id: ReadId, channel_id: Channel, signal: Signal
    ):
        self.read_id = read_id
        self.channel_id = channel_id

        self_group = self.add_to_group(parent_group)

        self.channel_id = channel_id.add_to_group(self_group)
        self.signal = signal.add_to_group(self_group)

    def name(self) -> str:
        return self.read_id


@dataclass
class SegmentationMeta(HDF5_GroupSerialableDataclass):
    segementer: str
    segementer_version: str
    capture_criteria: FilterSet
    terminal_captures_only: bool
    open_channel_prior_mean: float
    open_channel_prior_stdv: float
    good_channels: HDF5_DatasetSerializable
    capture_windows: CaptureWindows


class CaptureFile(BaseFile):
    def __init__(
        self, capture_filepath: PathLikeOrString, mode: str = "r", logger: Logger = getLogger()
    ):
        # TODO: Black doesn't like the formatting here, can't figure out why.
        # fmt: off
        """Capture file.

        Parameters
        ----------
        capture_filepath : PathLikeOrString
            Path to the capture file. Capture files are the result of running
            `poretitioner segment` on a bulk file.
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
            Capture file couldn't be opened
            (e.g. didn't exist, OS Resource temporarily unavailable).
            Details in message.
        ValueError
            Capture file had validation errors, details in message.
        """
        # TODO: Black doesn't like the formatting here, can't figure out why.
        # fmt: on
        logger.debug(f"Creating capture file at {capture_filepath} in mode ({mode})")
        super().__init__(capture_filepath, mode=mode, logger=logger)

        # Creates /Filters
        self.f5.require_group(FILTER_PATH.ROOT)

        # Creates /Classification
        self.f5.require_group(CLASSIFICATION_PATH.ROOT)

        if not self.filepath.exists():
            error_msg = f"Capture fast5 file does not exist at path: {self.filepath}. Make sure the capture file is in this location."
            raise OSError(error_msg)

    # @property
    # TODO: get the sub run info stored from initialization
    # def sub_run(self):
    #     self.f5[CAPTURE_PATH.CONTEXT_TAGS].attrs

    def initialize_from_bulk(
        self,
        bulk_f5: BulkFile,
        segment_config: SegmentConfiguration,
        capture_criteria: Optional[FilterSet] = None,
        sub_run: Optional[SubRun] = None,
        log: Logger = None,
    ):
        """Initializes the skeleton of a Capture fast5 file from the metadata of a Bulk fast5 file.
        This should be done before

        Parameters
        ----------
        bulk_f5 : BulkFile
            Bulk Fast5 file, generated from an Oxford Nanopore experiment.
        capture_criteria : Optional[FilterSet]
            Set of filters that define what signals could even potentially be a capture, by default None.
        segment_config : SegmentConfiguration
            General configuration for the segmenter.
        sub_run : Optional[SubRun], optional
            [description], by default None

        Raises
        ------
        ValueError
            [description]
        """
        # Only supporting range capture_criteria for now (MVP). This should be expanded to all FilterPlugins: https://github.com/uwmisl/poretitioner/issues/67

        # Referencing spec v0.1.1

        # /Meta/context_tags
        capture_context_tags_group = self.f5.require_group(CAPTURE_PATH.CONTEXT_TAGS)
        bulk_context_tags = bulk_f5.context_tags_group
        bulk_f5_fname = bulk_f5.filename
        context_tags_capture = ContextTagsCapture.from_context_tags_bulk_group(bulk_f5_fname, bulk_context_tags)
        capture_context_tags_group = context_tags_capture.add_to_group(
            capture_context_tags_group.parent, log=log
        )

        # capture_context_tags_group.attrs.create(key, value, dtype=hdf5_dtype(value))

        bulk_f5_fname = bulk_f5.filename

        # capture_context_tags_group.attrs.create(
        #     "bulk_filename", bulk_f5_fname, dtype=hdf5_dtype(bulk_f5_fname)
        # )

        # sampling_frequency = bulk_f5.sampling_rate
        # capture_context_tags_group.attrs.create("sample_frequency", sampling_frequency, dtype=hdf5_dtype(sampling_frequency))

        # /Meta/tracking_id
        capture_tracking_id_group = self.f5.require_group(CAPTURE_PATH.TRACKING_ID)
        for key, value in bulk_f5.tracking_id_group.get_attrs().items():
            capture_tracking_id_group.create_attr(key, value)

        # /
        if sub_run is not None:
            subrun_group = HDF5_Group(self.f5.require_group(CAPTURE_PATH.SUB_RUN))
            sub_run.add_to_group(subrun_group.parent, log=log)
            # id = sub_run.sub_run_id
            # offset = sub_run.sub_run_offset
            # duration = sub_run.sub_run_duration

            # capture_tracking_id_group.attrs.create(
            #     "sub_run_id", id, dtype=hdf5_dtype(id)
            # )
            # capture_tracking_id_group.attrs.create("sub_run_offset", offset)
            # capture_tracking_id_group.attrs.create("sub_run_duration", duration)

        # /Meta/Segmentation
        # TODO: define config param structure : https://github.com/uwmisl/poretitioner/issues/27
        # config = {"param": "value",
        #           "capture_criteria": {"f1": (min, max), "f2: (min, max)"}}
        capture_segmentation_group = self.f5.require_group(CAPTURE_PATH.SEGMENTATION)
        capture_windows_group = self.f5.require_group(CAPTURE_PATH.CAPTURE_WINDOWS)

        segmenter_name = get_application_info().name
        version = get_application_info().data_schema_version
        good_channels = [1]  # TODO: Pass in good channels
        filters = HDF5_FilterSet(capture_criteria)

        SEGMENT_METADATA = SegmentationMeta(
            segmenter_name,
            version,
            filters,
            segment_config.terminal_capture_only,
            segment_config.open_channel_prior_mean,
            segment_config.open_channel_prior_stdv,
            good_channels,
            capture_windows_group,
        )

        SEGMENT_METADATA.add_to_group(capture_segmentation_group, log=log)
        # print(__name__)

        # capture_segmentation_group.attrs.create(
        #     "segmenter", __name__, dtype=hdf5_dtype(segmenter_name)
        # )
        # capture_segmentation_group.attrs.create(
        #     "segmenter_version", version, dtype=hdf5_dtype(version)
        # )
        # filter_group = self.f5.require_group(CAPTURE_PATH.CAPTURE_CRITERIA)
        # context_id_group = self.f5.require_group(CAPTURE_PATH.CONTEXT_ID)
        # capture_windows_group = self.f5.require_group(CAPTURE_PATH.CAPTURE_WINDOWS)

        # Only supporting range capture_criteria for now (MVP): https://github.com/uwmisl/poretitioner/issues/67
        if segment_config is None:
            raise ValueError("No segment configuration provided.")
        else:
            self.log.info(f"Saving Segment config: {segment_config!s}")
            for key, value in vars(segment_config).items():
                try:
                    save_value = json.dumps(value)
                except TypeError:
                    # In case the object isn't easily serializable
                    save_value = json.dumps({k: v.__dict__ for k, v in value.items()})
                context_id_group.create_dataset(key, data=save_value)

            for name, my_filter in capture_criteria.items():
                filter_plugin = my_filter.plugin
                # `isinstance` is an anti-pattern, pls don't use in production.
                if isinstance(filter_plugin, RangeFilter):
                    maximum = filter_plugin.maximum
                    minimum = filter_plugin.minimum

                    self.log.debug(
                        f"Setting capture criteria for {name}: ({minimum}, {maximum})"
                    )

                    filter_group.create_dataset(name, data=(minimum, maximum))
            pass
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
        frequency.

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
    def reads(self, root: str = FAST5_ROOT) -> list[ReadId]:
        root_group = self.f5.require_group(root)
        potential_reads = [] if not root_group else root_group.keys()
        reads = [read for read in potential_reads if self.is_read(read)]
        return reads

    def filter(self, filter_set: FilterSet) -> set[ReadId]:
        # First, check whether this exact filter set exists in the file already

        filter_path = self.fast5.get(FILTER_PATH.ROOT)

        # If not, create it and write the result

        # If so, log that it was a found and return the result

    @property
    def filtered_reads(self):
        # TODO: Implement filtering here  to only return reads that pass a filter. https://github.com/uwmisl/poretitioner/issues/67
        return self.reads

    def filter_sets(self):
        pass

    def is_read(self, group: HDF5_Group) -> bool:
        """Whether this group points to a capture read.

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

        has_channel_path_group = issubclass(group.get(KEY.CHANNEL_ID, getclass=True), h5py.Group)
        has_signal_group = issubclass(group.get(KEY.SIGNAL, getclass=True), h5py.Dataset)
        return has_channel_path_group and has_signal_group

    def get_fractionalized_read(
        self, read_id: ReadId, start: Optional[int] = None, end: Optional[int] = None
    ) -> FractionalizedSignal:
        """Gets the fractionalized signal from this read.

        Parameters
        ----------
        read_id : ReadId
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
        signal_path = ChannelCalibration(read_id)
        channel_path = channel_path_for_read_id(read_id)
        open_channel_pA = self.f5[str(signal_path)].attrs[KEY.OPEN_CHANNEL_PA]
        calibration = self.get_channel_calibration_for_read(read_id)
        raw_signal: RawSignal = self.f5.get(str(signal_path))[start:end]
        channel_number = self.f5.get(str(channel_path)).attrs["channel_number"]
        fractionalized = FractionalizedSignal(
            raw_signal,
            channel_number,
            calibration,
            open_channel_pA,
            read_id=read_id,
            do_conversion=True,
        )

        return fractionalized

    def get_channel_calibration_for_read(self, read_id: ReadId) -> ChannelCalibration:
        """Retrieve the channel calibration for a specific read in a segmented fast5 file (i.e. CaptureFile).
        This is used for properly scaling values when converting raw signal to actual units.

        Note: using UK spelling of digitization for consistency w/ file format

        Parameters
        ----------
        read_id : ReadId
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

    def get_capture_metadata_for_read(self, read_id: ReadId) -> CaptureMetadata:
        """Retrieve the capture metadata for given read.

        Parameters
        ----------
        read_id : ReadId
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

    def write_capture(self, raw_signal: RawSignal, metadata: CaptureMetadata, log: Logger = None):
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

        channel_id = Channel(
            metadata.channel_number,
            metadata.calibration,
            metadata.open_channel_pA,
            metadata.sampling_rate,
        )
        # signal = HDF5_Signal.__new__(HDF5_Signal, raw_signal)
        signal = HDF5_Signal(
            raw_signal,
            metadata.start_time_bulk,
            metadata.start_time_local,
            metadata.duration,
            metadata.voltage_threshold,
            metadata.open_channel_pA,
            read_id,
            metadata.ejected,
            metadata.channel_number,
        )
        read = Read(read_id, channel_id, signal)
        read.add_to_group(f5, log=log)

        # signal_path = str(signal_path_for_read_id(read_id))
        # f5[signal_path] = raw_signal
        # f5[signal_path].attrs["read_id"] = read_id
        # f5[signal_path].attrs["start_time_bulk"] = metadata.start_time_bulk
        # f5[signal_path].attrs["start_time_local"] = metadata.start_time_local
        # f5[signal_path].attrs["duration"] = metadata.duration
        # f5[signal_path].attrs["ejected"] = metadata.ejected
        # f5[signal_path].attrs["voltage"] = metadata.voltage_threshold
        # f5[signal_path].attrs[KEY.OPEN_CHANNEL_PA] = metadata.open_channel_pA

        # channel_path = str(channel_path_for_read_id(read_id))
        # f5.require_group(channel_path)
        # f5[channel_path].attrs["channel_number"] = metadata.channel_number
        # f5[channel_path].attrs["digitisation"] = metadata.calibration.digitisation
        # f5[channel_path].attrs["range"] = metadata.calibration.range
        # f5[channel_path].attrs["offset"] = metadata.calibration.offset
        # f5[channel_path].attrs["sampling_rate"] = metadata.sampling_rate
        # f5[channel_path].attrs[KEY.OPEN_CHANNEL_PA] = metadata.open_channel_pA

    def write_capture_windows(self, capture_windows: WindowsByChannel):
        pass
