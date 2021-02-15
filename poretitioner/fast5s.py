"""
===================
fast5.py
===================

Classes for reading, writing and validating fast5 files.
"""

from os import PathLike
from pathlib import Path, PurePosixPath
from typing import List, NewType, Optional, Union

import h5py

from .logger import Logger, getLogger
from .signals import (
    CaptureMetadata,
    ChannelCalibration,
    FractionalizedSignal,
    RawSignal,
    VoltageSignal,
)

from .utils.classify import (
    NULL_CLASSIFICATION_RESULT,
    ClassificationResult,
    ClassifierConfiguration,
    ClassifierDetails,
    NullClassificationResult,
)

__all__ = ["BulkFile", "CaptureFile", "channel_path_for_read_id", "signal_path_for_read_id"]

FAST5_ROOT = "/"
CHANNEL_ID_KEY = "channel_id"


def add_signal_to_path(base: PathLike) -> PathLike:
    # We're using PurePosixPath, instead of Path, to guarantee that the path separator will be "/" (instead of using the operating system default)
    path_with_signal = PurePosixPath(base, "Signal")
    return path_with_signal


def add_channel_id_to_path(base: PathLike) -> PathLike:
    # We're using PurePosixPath, instead of Path, to guarantee that the path separator will be "/" (instead of using the operating system default)
    path_with_channel_id = PurePosixPath(base, "channel_id")
    return path_with_channel_id


def is_read(f5: h5py.File, path: str) -> bool:
    # Reads should have Signal and channelId groups.
    # If a path has both of theses, we consider the group a signal
    channel_path = str(add_channel_id_to_path(path))
    signal_path = str(add_signal_to_path(path))
    return f5.get(channel_path) and f5.get(signal_path)


class BaseFile:
    def __init__(self, filepath: Union[str, PathLike], logger: Logger = getLogger()):
        """Base class for interfacing with Fast5 files. This class should not be instantiated directly, instead it should be subclassed.
        Most of the time, nanopore devices/software write data in an HDFS [1] file format called Fast5.
        We expect a certain format for these files, and write our own.

        [1] - https://support.hdfgroup.org/HDF5/whatishdf5.html

        Parameters
        ----------
        filepath : [type]
            [description]
        logger : Logger, optional
            [description], by default getLogger()

        Raises
        ------
        OSError
            Bulk file couldn't be opened (e.g. didn't exist, OS 'Resource temporarily unavailable'). Details in message.
        ValueError
            Bulk file had validation errors, details in message.
        """

        self.filepath = Path(filepath).expanduser().resolve()
        self.f5 = h5py.File(self.filepath, "r")
        self.filename = self.f5.filename
        self.log = logger

    def __enter__(self):
        return self.f5.__enter__()

    def __exit__(self):
        self.f5.__exit__()

    def get_channel_calibration_for_path(self, path: str) -> ChannelCalibration:
        """Gets the channel calibration

        Parameters
        ----------
        path : str
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

    def __init__(self, bulk_filepath: PathLike, logger: Logger = getLogger()):
        """Interfaces with Bulk fast5 files. The Bulk fast5 refers to the file format generated by Oxford Nanopore MinKnow devices.

        Parameters
        ----------
        bulk_filepath : PathLike
            Absolute path to the bulk fast 5 file.
        logger : Logger, optional
            Logger to use, by default getLogger()

        Raises
        ------
        OSError
            Bulk file couldn't be opened (e.g. didn't exist, OS 'Resource temporarily unavailable'). Details in message.
        ValueError
            Bulk file had validation errors, details in message.
        """
        super().__init__(bulk_filepath, logger=logger)
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
            # TODO: Katie Q: Why the [2:-1] at the end?
            run_id = str(tracking_id.attrs["run_id"])[2:-1]
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

        Parameters
        ----------
        f5 : h5py.File
            Fast5 file, open for reading using h5py.File.

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
        f5 : h5py.File
            Fast5 file, open for reading using h5py.File.

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
        voltages = metadata["bias_voltage"][start:end] * bias_voltage_multiplier
        return voltages


class CaptureFile(BaseFile):
    def __init__(self, capture_filepath: PathLike, logger: Logger = getLogger()):
        """Capture file.

        Parameters
        ----------
        capture_filepath : PathLike
            Path to the capture file. Capture files are the result of running `poretitioner segment` on a bulk file.
        logger : Logger, optional
            [description], by default getLogger()

        Raises
        ------
        OSError
            Capture file couldn't be opened (e.g. didn't exist, OS 'Resource temporarily unavailable'). Details in message.
        ValueError
            Capture file had validation errors, details in message.
        """
        super().__init__(capture_filepath, logger=logger)
        if not self.filepath.exists():
            error_msg = f"Capture fast5 file does not exist at path: {self.filepath}. Make sure the bulk file is in this location."
            raise OSError(error_msg)
        self.validate(self.filepath, logger)

    def validate(self, capture_filepath: PathLike, log: Logger = getLogger()):
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
            pass  # Try checking the Meta group as a fallback.

        try:
            sample_rate = int(self.f5[sample_rate_path].attrs[sample_rate_key])
        except KeyError:
            error_msg = f"Sampling rate not present in bulk file '{self.f5.filename}'. Make sure a sampling frequency is specified either at '${sample_frequency_path}' with attribute '{sample_frequency_key}', or as a fallback, '{sample_rate_path}' with attribute '{sample_rate_key}'"
            self.log.error(error_msg)
            raise ValueError(error_msg)

        rate = sample_frequency if sample_frequency else sample_rate
        return rate

    @property
    def reads(self, root: str = FAST5_ROOT) -> List[str]:
        potential_read = [] if not self.f5.get(root) else self.f5.get(root).keys()
        reads = [read for read in potential_read if is_read(self.f5, read)]
        return reads

    @property
    def filtered_reads(self):
        # TODO: Implement filtering here  to only return reads that pass a filter. https://github.com/uwmisl/poretitioner/issues/67
        self.reads

    def fractionalized_read(
        self, read_id: str, start: Optional[int] = None, end: Optional[int] = None
    ):
        signal_path = signal_path_for_read_id(read_id)
        channel_path = channel_path_for_read_id(read_id)
        open_channel_pA = self.f5[signal_path].attrs["open_channel_pA"]
        calibration = self.get_channel_calibration_for_read(read_id)
        raw_signal = self.f5.get(signal_path)[start:end]
        channel_number = self.f5.get(channel_path).attrs["channel_number"]

        fractionalized = FractionalizedSignal(
            raw_signal,
            channel_number,
            calibration,
            open_channel_pA,
            do_conversion=True,
            read_id=read_id,
        )
        return fractionalized

    def get_channel_calibration_for_read(self, read_id: str) -> ChannelCalibration:
        """Retrieve the channel calibration for a specific read in a segmented fast5 file (i.e. CaptureFile).
        This is used for properly scaling values when converting raw signal to actual units.

        Note: using UK spelling of digitization for consistency w/ file format

        Parameters
        ----------
        f5 : h5py.File
            Fast5 file, open for reading using h5py.File.
        read_id : str
            Read id to retrieve raw signal. Can be formatted as a path ("read_xxx...")
            or just the read id ("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").

        Returns
        -------
        ChannelCalibration
            Channel calibration Offset, range, and digitisation values.
        """
        channel_path = str(channel_path_for_read_id(read_id))
        calibration = self.get_channel_calibration_for_path(channel_path)
        return calibration

    def get_capture_metadata_for_read(self, read_id: str) -> CaptureMetadata:
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
        open_channel_pA = self.f5[signal_path].attrs["open_channel_pA"]

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

    def write_capture(
        self, capture_f5_filepath: PathLike, raw_signal: RawSignal, metadata: CaptureMetadata
    ):
        """Write a single capture to the specified capture fast5 file (which has
        already been created via create_capture_fast5()).

        Parameters
        ----------
        capture_f5_filepath : PathLike
            Filename of the capture fast5 file to be augmented.
        raw_signal : RawSignal
            Time series of nanopore current values (in units of pA).
        metadata: CaptureMetadata
            Details about this capture.
        """
        path = Path(capture_f5_filepath)
        save_directory = Path(path.parent)
        if not save_directory.exists():
            raise IOError(f"Path to capture file location does not exist: {save_directory}")
            CaptureMetadata
        read_id = metadata.read_id
        f5 = self.f5
        signal_path = f"read_{read_id}/Signal"
        f5[signal_path] = raw_signal
        f5[signal_path].attrs["read_id"] = read_id
        f5[signal_path].attrs["start_time_bulk"] = metadata.start_time_bulk
        f5[signal_path].attrs["start_time_local"] = metadata.start_time_local
        f5[signal_path].attrs["duration"] = metadata.duration
        f5[signal_path].attrs["ejected"] = metadata.ejected
        f5[signal_path].attrs["voltage"] = metadata.voltage_threshold
        f5[signal_path].attrs["open_channel_pA"] = metadata.open_channel_pA

        channel_path = channel_path_for_read_id(read_id)
        f5.create_group(channel_path)
        f5[channel_path].attrs["channel_number"] = metadata.channel_number
        f5[channel_path].attrs["digitisation"] = metadata.calibration.digitisation
        f5[channel_path].attrs["range"] = metadata.calibration.rng
        f5[channel_path].attrs["offset"] = metadata.calibration.offset
        f5[channel_path].attrs["sampling_rate"] = metadata.sampling_rate
        f5[channel_path].attrs["open_channel_pA"] = metadata.open_channel_pA


class ClassifierFile(CaptureFile):
    def __init__(
        self,
        capture_filepath: PathLike,
        classifier_details: ClassifierDetails,
        logger: Logger = getLogger(),
    ):
        super().__init__(capture_filepath, logger=logger)

        self.classifier_path = PurePosixPath("/", "Classification", classifier_details.model)
        # self.results_path = str(PurePosixPath("/", "Classification", classifier_run_name, read_id))
        f5 = self.f5

        if self.classifier_path not in self.f5:
            self.f5.create_group(self.classifier_path)
        self.f5[self.classifier_path].attrs["model"] = classifier_details.model
        self.f5[self.classifier_path].attrs["model_version"] = classifier_details.model_version
        self.f5[self.classifier_path].attrs["model_file"] = classifier_details.model_file
        self.f5[self.classifier_path].attrs[
            "classification_threshold"
        ] = classifier_details.classification_threshold

    def result_for_read(self, read_id: str) -> ClassificationResult:
        result_path = str(PurePosixPath(self.classifier_path, read_id))
        f5 = self.f5
        result = NULL_CLASSIFICATION_RESULT
        if result_path not in f5:
            self.log.info(
                f"Read {read_id} has not been classified yet, or result"
                f"is not stored at {result_path} in file {f5.filename}."
            )
            return result
        predict_class = f5[result_path].attrs["best_class"]
        probability = f5[result_path].attrs["best_score"]
        assigned_class = f5[result_path].attrs["assigned_class"]
        result = ClassificationResult(predict_class, probability, assigned_class)
        return result

    def __get_results_path__(self, read_id: str):
        results_path = PurePosixPath(self.classifier_path, read_id)
        return results_path

    def add_model_results(self,):
        pass

    def write_details(self, classifier_details: ClassifierDetails):
        """Write metadata about the classifier that doesn't need to be repeated for
        each read.

        Parameters
        ----------
        f5 : h5py.File
            Opened fast5 file in a writable mode.
        classifier_config : dict
            Subset of the configuration parameters that belong to the classifier.
        results_path : str
            Where the classification results will be stored in the f5 file.
        """
        results_path = self.classifier_path
        if results_path not in self.f5:
            self.f5.create_group(results_path)
        self.f5[results_path].attrs["model"] = classifier_details.model
        self.f5[results_path].attrs["model_version"] = classifier_details.model_version
        self.f5[results_path].attrs["model_file"] = classifier_details.model_file
        self.f5[results_path].attrs[
            "classification_threshold"
        ] = classifier_details.classification_threshold

    def write_result(self, read_id: str, result: ClassificationResult):
        results_path = str(PurePosixPath(self.classifier_path, read_id))
        if results_path not in self.f5:
            self.f5.create_group(results_path)
        self.f5[results_path].attrs["best_class"] = result.predicted
        self.f5[results_path].attrs["best_score"] = result.probability
        self.f5[results_path].attrs["assigned_class"] = (
            result.assigned_class if result.passed_classification else -1
        )


class ClassifiedCapture:
    def __init__(self) -> None:
        pass


def channel_path_for_read_id(read_id: str) -> PathLike:
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
    read_id_path = PurePosixPath("/", read_id)
    channel_path = add_channel_id_to_path(read_id_path)

    if "read" in read_id:
        channel_path = PurePosixPath("/", read_id, CHANNEL_ID_KEY)
    else:
        channel_path = PurePosixPath("/", f"read_{read_id}", CHANNEL_ID_KEY)
    # channel_path = str(channel_path)
    return channel_path


def signal_path_for_read_id(read_id: str) -> PathLike:
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
        signal_path = PurePosixPath("/", read_id, "Signal")
    else:
        signal_path = PurePosixPath("/", f"read_{read_id}", "Signal")
    # signal_path = str(signal_path)
    return signal_path
