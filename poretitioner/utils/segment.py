"""
==========
segment.py
==========

This module contains functionality for segmenting nanopore captures from
bulk fast5s.

"""
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
# TODO: Pipe through filtering https://github.com/uwmisl/poretitioner/issues/43 https://github.com/uwmisl/poretitioner/issues/68
# from . import filter
from typing import Dict, Iterable, List, Optional

import dask.bag as db
import h5py
import numpy as np
from dask.diagnostics import ProgressBar

from .. import application_info, fast5adapter, logger
from ..fast5s import BulkFile, CaptureFile
from ..signals import (
    Capture,
    CaptureMetadata,
    Channel,
    ChannelCalibration,
    FractionalizedSignal,
    PicoampereSignal,
    RawSignal,
    SignalMetadata,
    digitize_current,
    find_open_channel_current,
)
from .core import Window, find_windows_below_threshold

ProgressBar().register()

app_info = application_info.get_application_info()

__version__ = app_info.version
__name__ = app_info.name

__all__ = ["segment"]


def find_captures(
    signal_pA: PicoampereSignal,
    signal_threshold_frac: float,
    open_channel_pA_calculated: float,
    terminal_capture_only: bool = False,
    filters={},
    delay=50,
    end_tol=0,
) -> List[Capture]:
    """For a single section of current (capture window, in units of pA), identify
    any captures within the window.

    Current is first converted from pA to a fractionalized (0, 1) range based on
    open channel current, and then any spans of current below the threshold
    are considered possible captures. These are further reduced by filters and
    whether the capture must be in the pore at the end of the window.

    Parameters
    ----------
    signal_pA : PicoampereSignal
        Time series of nanopore current values (in units of pA).
    signal_threshold_frac : float
        Threshold for the first pass of finding captures (in fractional
        current). (Captures are <= the threshold.)
    open_channel_pA_calculated : float
        Calculated estimate of the open channel current value.
    terminal_capture_only : bool, optional
        Only return the final capture in the window, and only if it remains
        captured until the end of the window (to be ejected), by default False
    filters : dict, optional
        Keys are strings matching the supported filters, values are a tuple
        giving the endpoints of the valid range. E.g. {"mean": (0.1, 0.5)}
        defines a filter such that 0.1 <= mean(capture) <= 0.5., default {}
    delay : int, optional
        Translocation delay. At high sampling rates, there may be some data
        points at the beginning of the capture that represent the initial
        translocation of the molecule. The beginning of the capture will be
        trimmed by _delay_ points, by default 50.
    end_tol : int, optional
        If terminal_capture_only is True, specify close to the end of the
        window the last capture must be in order to count. A value of 0 means
        the end of the last capture must exactly align with the end of the
        window.

    Returns
    -------
    List[Capture]
        List of captures in the given window of signal.
    """

    # Convert to frac current
    frac_current = signal_pA.to_fractionalized()
    del signal_pA
    # Apply signal threshold & get list of captures
    capture_windows = find_windows_below_threshold(
        frac_current, signal_threshold_frac
    )  # find_segments_below_threshold(frac_current, signal_threshold_frac)

    captures = [
        Capture(
            frac_current[window.start : window.end],
            window,
            signal_threshold_frac,
            open_channel_pA_calculated,
        )
        for window in capture_windows
    ]
    # If terminal_capture_only, reduce list of captures to only last
    if terminal_capture_only:
        if len(captures) > 1 and captures[-1].ejected:
            captures = [captures[-1]]
        else:
            captures = []

    last_capture = captures[-1]
    if delay > 0:
        for i, capture in enumerate(captures):
            capture_start, capture_end = capture.window
            if capture.window.duration > delay:
                delayed_window = Window(capture_start + delay, capture_end)
                captures[i] = Capture(
                    capture.signal,
                    delayed_window,
                    signal_threshold_frac,
                    open_channel_pA_calculated,
                )

    # Apply filters to remaining capture(s)
    filtered_captures = []
    capture_start, capture_end = last_capture.window
    # TODO: Pipe through filtering https://github.com/uwmisl/poretitioner/issues/43 https://github.com/uwmisl/poretitioner/issues/68
    # filtered = [capture for capture in captures if filter.apply_feature_filters(frac_current[capture_start:capture_end], filters)]
    filtered = [
        capture
        for capture in captures
        if filter.apply_feature_filters(frac_current[capture_start:capture_end], filters)
    ]
    return filtered


def find_captures_dask_wrapper(
    capture: Capture, terminal_capture_only=False, filters={}, delay=50, end_tol=0
):
    """Wrapper for find_captures since dask bag can only take one arg as input,
    plus kwargs. See find_captures() for full description.

    Parameters
    ----------
    capture : Capture signal
    terminal_capture_only : bool, optional
        Only return the final capture in the window, and only if it remains
        captured until the end of the window (to be ejected), by default False
    filters : dict, optional
        Keys are strings matching the supported filters, values are a tuple
        giving the endpoints of the valid range. E.g. {"mean": (0.1, 0.5)}
        defines a filter such that 0.1 <= mean(capture) <= 0.5., default {}
    delay : int, optional
        Translocation delay. At high sampling rates, there may be some data
        points at the beginning of the capture that represent the initial
        translocation of the molecule. The beginning of the capture will be
        trimmed by _delay_ points, by default 50.
    end_tol : int, optional
        If terminal_capture_only is True, specify close to the end of the
        window the last capture must be in order to count. A value of 0 means
        the end of the last capture must exactly align with the end of the
        window.
    Returns
    -------
    List[Capture]
        List of captures in the given window of signal.
    """
    return find_captures(
        capture.signal,
        capture.signal_threshold_frac,
        capture.open_channel_pA_calculated,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )


def create_capture_fast5(
    bulk_f5_fname, capture_f5_fname, config, overwrite=False, sub_run=(None, None, None)
):
    """Prepare a fast5 file to contain the results of segmentation (capture
    fast5 file). The file does not contain raw data yet, just a skeleton.

    Parameters
    ----------
    bulk_f5_fname : str
        Filename of the bulk fast5, to be segmented into captures.
    capture_f5_fname : str
        Filename of the capture fast5 file to be created/written. Directory
        containing this file must already exist.
    config : dict
        Configuration parameters for the segmenter.
        # TODO : document allowed values : https://github.com/uwmisl/poretitioner/issues/27
    overwrite : bool, optional
        Flag whether or not to overwrite capture_f5_fname, by default False
    sub_run : tuple, optional
        If the bulk fast5 contains multiple runs (shorter sub-runs throughout
        the data collection process), this can be used to record additional
        context about the sub run: (sub_run_id : str, sub_run_offset : int, and
        sub_run_duration : int). sub_run_id is the identifier for the sub run,
        sub_run_offset is the time the sub run starts in the bulk fast5,
        measured in #/time series points.

    Raises
    ------
    OSError
        Raised if the bulk fast5 file does not exist.
        Raised if the path to the capture fast5 file does not exist.
    """
    local_logger = logger.getLogger()
    if not os.path.exists(bulk_f5_fname):
        raise OSError(f"Bulk fast5 file does not exist: {bulk_f5_fname}")
    if overwrite:
        if os.path.exists(capture_f5_fname):
            os.remove(capture_f5_fname)
    else:
        try:
            assert not os.path.exists(capture_f5_fname)
        except AssertionError:
            raise FileExistsError()

    path, fname = os.path.split(capture_f5_fname)
    if not os.path.exists(path):
        raise OSError(f"Path for the new capture fast5 does not exist: {path}")

    with h5py.File(capture_f5_fname, "a") as capture_f5:
        with h5py.File(bulk_f5_fname, "r") as bulk_f5:
            ugk = bulk_f5.get("/UniqueGlobalKey")

            # Referencing spec v0.1.1
            # /Meta
            capture_f5.create_group("/Meta")

            # /Meta/context_tags
            attrs = ugk["context_tags"].attrs
            g = capture_f5.create_group("/Meta/context_tags")
            for k, v in attrs.items():
                g.attrs.create(k, v)
            g.attrs.create("bulk_filename", bulk_f5_fname, dtype=f"S{len(bulk_f5_fname)}")
            sample_frequency = bulk_f5.get("Meta").attrs["sample_rate"]
            g.attrs.create("sample_frequency", sample_frequency)

            # /Meta/tracking_id
            attrs = ugk["tracking_id"].attrs
            g = capture_f5.create_group("/Meta/tracking_id")
            for k, v in attrs.items():
                g.attrs.create(k, v)

            if sub_run is not None:
                sub_run_id, sub_run_offset, sub_run_duration = sub_run
                if sub_run_id is not None:
                    g.attrs.create("sub_run_id", sub_run_id, dtype=f"S{len(sub_run_id)}")
                if sub_run_offset is not None:
                    g.attrs.create("sub_run_offset", sub_run_offset)
                if sub_run_duration is not None:
                    g.attrs.create("sub_run_duration", sub_run_duration)

            # /Meta/Segmentation
            # TODO: define config param structure : https://github.com/uwmisl/poretitioner/issues/27
            # config = {"param": "value",
            #           "filters": {"f1": (min, max), "f2: (min, max)"}}
            g = capture_f5.create_group("/Meta/Segmentation")
            # print(__name__)
            g.attrs.create("segmenter", __name__, dtype=f"S{len(__name__)}")
            g.attrs.create("segmenter_version", __version__, dtype=f"S{len(__version__)}")
            g_filt = capture_f5.create_group("/Meta/Segmentation/filters")
            g_seg = capture_f5.create_group("/Meta/Segmentation/context_id")
            segment_config = config.get("segment")
            if segment_config is None:
                raise ValueError("No segment configuration provided.")
            for k, v in segment_config.items():
                if k == "filter":
                    for filt, filt_vals in v.items():
                        if len(filt_vals) == 2:
                            (min_filt, max_filt) = filt_vals
                            # Create compound dset for filters
                            local_logger.debug("filt types", type(min_filt), type(max_filt))
                            if min_filt is None:
                                min_filt = -1
                            if max_filt is None:
                                max_filt = -1
                            if min_filt == -1 and max_filt == -1:
                                continue
                            g_filt.create_dataset(filt, data=(min_filt, max_filt))
                else:
                    g_seg.create_dataset(k, data=v)


@dataclass
class PreppedCaptureCandidates:
    """ A series of captures ready to be processed in paralllel.

    Parameters
    ----------
    captures: List[Capture]
        List of Captures to be procssed.
    metadata: List[SignalMetadata]
        Metadata about the captures.
    run_id: str
        Unique identifier for this run.
    sampling_rate: int
        Sampling rate for the signal.
    """

    captures: List[Capture]
    metadata: List[SignalMetadata]
    run_id: str
    sampling_rate: int


def _prep_capture_windows(
    bulk_f5_fname: str,
    voltage_threshold: int,
    signal_threshold_frac: float,
    good_channels: List[int],
    open_channel_pA_calculated: float,
    f5_subsection_start: int = 0,
    f5_subsection_end: Optional[int] = None,
) -> PreppedCaptureCandidates:
    """Helper function to extract raw data from the bulk fast5 for segmentation.

    Parameters
    ----------
    bulk_f5_fname : str
        Filename of the bulk fast5, to be segmented into captures.
    f5_subsection_start : int
        Time point at which to start segmenting, by default 0
    f5_subsection_end : int, optional
        Time point at which to stop segmenting, by default None
    voltage_threshold : int
        Voltage must be at or below this value to be considered a capture window.
    signal_threshold_frac : float
        Threshold for the first pass of finding captures (in fractional
        current). (Captures are <= the threshold.)
    good_channels : List[int]
        Specifies the nanopore array channels to be segmented.
    open_channel_pA_calculated : float
        Calculated estimate of the open channel current value.

    Returns
    -------
    PreppedCaptureCandidates
        Segments of the raw signal representing capture windows, and metadata
        about these segments (channel number, window endpoints, offset, range,
        digitisation).

    Raises
    ------
    ValueError
        Raised when no voltage is present at or below the threshold.
        Raised when any channel in good_channels is not present in the bulk file.
    """
    local_logger = logger.getLogger()
    with BulkFile(bulk_f5_fname) as bulk_f5:
        local_logger.info(f"Reading in signals for bulk file: {bulk_f5_fname}")
        voltage = bulk_f5.get_voltage(start=f5_subsection_start, end=f5_subsection_end)
        local_logger.debug(f"voltage: {voltage}")

        run_id = bulk_f5.run_id
        sampling_rate = bulk_f5.sampling_rate

        local_logger.info("Identifying capture windows (via voltage threshold).")
        capture_windows = find_windows_below_threshold(voltage, voltage_threshold)
        if len(capture_windows) == 0:
            raise ValueError(f"No voltage at or below threshold ({voltage_threshold})")

        local_logger.debug(
            f"run_id: {run_id}, sampling_rate: {sampling_rate}, "
            f"#/capture windows: {len(capture_windows)}"
        )
        local_logger.debug("Prepping raw signals for parallel processing.")
        captures = []  # Input data to find_captures_dask_wrapper
        signal_metadata = []  # Metadata -- no need to pass through the segmenter
        for channel_number in good_channels:
            calibration = bulk_f5.get_channel_calibration(channel_number)

            raw_signal: RawSignal = bulk_f5.get_raw_signal(
                channel_number, start=f5_subsection_start, end=f5_subsection_end
            )
            pico = raw_signal.to_picoamperes()
            for capture_window in capture_windows:
                start, end = capture_window.start, capture_window.end

                capture = Capture(
                    pico[start:end],
                    capture_window,
                    signal_threshold_frac,
                    open_channel_pA_calculated,
                )
                metadata = SignalMetadata(
                    channel_number=channel_number, window=capture_window, calibration=calibration
                )

                captures.append(capture)
                signal_metadata.append(metadata)
        prepped = PreppedCaptureCandidates(captures, signal_metadata, run_id, sampling_rate)
        return prepped


def segment(
    bulk_f5_filepath, save_location, config: Dict, f5_subsection_start=None, f5_subsection_end=None
):
    """[summary]

    Parameters
    ----------
    bulk_f5_filepath : [type]
        [description]
    save_location : str
        Directory to save the segmentation results (results will often be saved to more than one file).
    config : Dict
        Segmentaiotn configuration # TODO: Pipe through the configuration here: https://github.com/uwmisl/poretitioner/issues/73
    f5_subsection_start : [type], optional
        [description], by default None
    f5_subsection_end : [type], optional
        [description], by default None
    """
    local_logger = logger.getLogger()

    local_logger.debug(
        f"Segmenting bulk file '{bulk_f5_filepath}' and saving results at location '{save_location}' "
    )
    local_logger.debug(
        f"Segmenting configuration '{bulk_f5_filepath}' and saving results at location '{save_location}' "
    )
    parallel_find_captures(
        bulk_f5_filepath,
        config,
        save_location=save_location,
        f5_subsection_start=f5_subsection_start,
        f5_subsection_end=f5_subsection_end,
    )


def parallel_find_captures(
    bulk_f5_fname, config, save_location=None, f5_subsection_start=None, f5_subsection_end=None
):
    """Identify captures within the bulk fast5 file in the specified range
    (from f5_subsection_start to f5_subsection_end.)

    If f5_subsection_{start|end} are not specified, the default is to use the
    range of current starting from the beginning and/or end of the bulk fast5.

    Note: config must be validated before this function.

    Parameters
    ----------
    bulk_f5_fname : str
        Filename of the bulk fast5, to be segmented into captures.
    save_location: Optional[str]
        Where to save the segmented captures. If not provided, use what's in the config.
    config : dict
        Configuration parameters for the segmenter.
        # TODO : document allowed values : https://github.com/uwmisl/poretitioner/issues/27 https://github.com/uwmisl/poretitioner/issues/73
    f5_subsection_start : int, optional
        Time point at which to start segmenting, by default None
    f5_subsection_end : int, optional
        Time point at which to stop segmenting, by default None
    overwrite : bool, optional
        If segmented files already exist, overwrite them if True, otherwise
        return None.

    Returns
    -------
    List[]
        Metadata about the captures. To be used for diagnostic purposes.

    Raises
    ------
    IOError
        If the desired output location does not exist (i.e., save_location) in
        the config, raise IOError.
    """

    local_logger = logger.getLogger()
    # TODO: Update with actual configuring https://github.com/uwmisl/poretitioner/issues/73
    n_workers = config["compute"]["n_workers"]
    assert type(n_workers) is int
    voltage_threshold = config["segment"]["voltage_threshold"]
    signal_threshold_frac = config["segment"]["signal_threshold"]
    delay = config["segment"]["translocation_delay"]
    open_channel_prior_mean = config["segment"]["open_channel_prior_mean"]
    good_channels = config["segment"]["good_channels"]
    end_tol = config["segment"]["end_tol"]
    terminal_capture_only = config["segment"]["terminal_capture_only"]
    filters = config["filters"]["base filter"]
    save_location = (
        save_location if save_location else config["output"]["capture_f5_dir"]
    )  # TODO: Verify exists; don't create (handle earlier)
    n_per_file = config["output"]["captures_per_f5"]
    # if f5_subsection_start is None:
    #     f5_subsection_start = 0

    if not os.path.exists(save_location):
        raise IOError(f"Path to capture file location does not exist: {save_location}")

    prepped_captures: PreppedCaptureCandidates = _prep_capture_windows(
        bulk_f5_fname,
        voltage_threshold,
        signal_threshold_frac,
        good_channels,
        open_channel_prior_mean,
        f5_subsection_start=f5_subsection_start,
        f5_subsection_end=f5_subsection_end,
    )

    local_logger.debug("Loading up the bag with signals.")
    bag = db.from_sequence(prepped_captures.captures, npartitions=64)
    capture_map = bag.map(
        find_captures_dask_wrapper,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    local_logger.info("Beginning segmentation.")
    captures = capture_map.compute(num_workers=n_workers)
    local_logger.debug(f"Captures (1st 10): {captures[:10]}")

    context = prepped_captures.metadata
    run_id = prepped_captures.run_id
    sampling_rate = prepped_captures.sampling_rate

    # Channel Calibration is repeated in the context, even though the calibration doesn't change per capture. Consider optimizing this deadweight.
    assert len(captures) == len(context)

    # Write captures to fast5
    n_in_file = 0
    file_no = 1
    capture_metadata = []
    # for (
    #     i,
    #     ((captures_in_window, open_channel_pA), window_context, (window_raw, _, _)),
    # ) in enumerate(zip(captures, context, raw_signals)):

    def get_capture_metadata(
        read_id: str,
        sampling_rate: int,
        voltage_threshold: int,
        capture: Capture,
        metadata: SignalMetadata,
        f5_subsection_start: int = 0,
    ) -> CaptureMetadata:
        """Summarizes a capture in metadata.

        Parameters
        ----------
        read_id : str
            Uniquely identifies the read from which this capture was identified.
        capture : Capture
            The capture in question.
        metadata : SignalMetadata
            Metadata around the signal that generated the capture.

        Returns
        -------
        CaptureMetadata
            Metadata for the capture.
        """
        window_start = metadata.window.start

        capture_start = capture.window.start

        channel_number = capture.signal.channel_number

        start_time_local = (
            capture_start + window_start
        )  # Start time relative to the start of the bulk f5 subsection
        start_time_bulk = (
            start_time_local + f5_subsection_start
        )  # relative to the start of the entire bulk f5
        capture_duration = capture.duration
        ejected = capture.ejected
        local_logger.debug(f"Capture duration: {capture_duration}")

        open_channel_pA = capture.open_channel_pA_calculated
        calibration = capture.signal.calibration
        capture_metadata = CaptureMetadata(
            read_id,
            start_time_bulk,
            start_time_local,
            capture_duration,
            ejected,
            voltage_threshold,
            open_channel_pA,
            channel_number,
            calibration,
            sampling_rate,
        )
        return capture_metadata

    for capture, metadata in zip(captures, context):
        for capture in capture:
            read_id = str(uuid.uuid4())
            start, end = capture.window.start, capture.window.end
            capture_metadata = get_capture_metadata(
                read_id,
                sampling_rate,
                voltage_threshold,
                capture,
                metadata,
                f5_subsection_start=f5_subsection_start,
            )

            if n_in_file >= n_per_file:
                n_in_file = 0
                file_no += 1

            capture_f5_filepath = Path(save_location, f"{run_id}_{file_no}.fast5")
            n_in_file += 1

            picoamperes: PicoampereSignal = capture.signal.to_picoamperes()
            raw_signal = picoamperes.to_raw()[start:end]
            local_logger.debug(f"Length of raw signal : {len(picoamperes)}")

            local_logger.debug(f"Writing to file: {capture_f5_filepath}")
            write_capture_to_fast5(capture_f5_filepath, raw_signal, capture_metadata)

            write_capture_to_fast5(
                Path(save_location, f"{run_id}_{file_no}.hdf"), raw_signal, capture_metadata
            )
            local_logger.debug(f"\tWritten!")

    return capture_metadata


def sort_capture_windows_by_channel(signal_metadata):
    # Initialize dictionary of {channel: [windows]}
    try:
        channels = np.unique(signal_metadata[:, 0])
    except TypeError:
        raise TypeError("signal_metadata must be an array.")
    windows_by_channel = {}
    for channel in channels:
        windows_by_channel[channel] = []

    # Populate dictionary values
    for row in signal_metadata:
        channel = row[0]
        windows_by_channel[channel].append(row)

    # Sort the windows within each channel by time
    for channel, captures in windows_by_channel.items():
        captures = np.array(captures)
        windows_by_channel[channel] = captures[captures[:, 1].argsort()]
    return windows_by_channel


def write_capture_windows_to_fast5(capture_f5_fname, signal_metadata):
    # Check to make sure the path exists
    path, fname = os.path.split(capture_f5_fname)
    if not os.path.exists(path):
        raise IOError(f"Path to capture file location does not exist: {path}")

    with h5py.File(capture_f5_fname, "a") as f5:
        base_path = "/Meta/Segmentation/capture_windows"
        # Sort the windows by channel and write to file
        # signal_metadata = [[channel_no, capture_window, offset, rng, digi], ...]
        signal_metadata = np.array(signal_metadata, dtype=object)
        windows_by_channel = sort_capture_windows_by_channel(signal_metadata)
        for channel, signal_meta in windows_by_channel.items():
            path = f"{base_path}/Channel_{channel}"
            capture_windows = []
            for meta in signal_meta:
                channel_no, capture_window, offset, rng, digi = meta
                capture_windows.append(capture_window)
            f5[path] = capture_windows


def write_capture_to_fast5(
    capture_f5_fname: str, raw_signal: RawSignal, metadata: CaptureMetadata
):
    """Write a single capture to the specified capture fast5 file (which has
    already been created via create_capture_fast5()).

    Parameters
    ----------
    filename : str
        Filename of the capture fast5 file to be augmented.
    raw_signal : RawSignal
        Time series of nanopore current values (in units of pA).
    metadata: CaptureMetadata
        Details about this capture.
    """
    path = Path(capture_f5_fname)
    save_directory = Path(path.parent)
    if not save_directory.exists():
        raise IOError(f"Path to capture file location does not exist: {save_directory}")
        CaptureMetadata
    read_id = metadata.read_id
    with h5py.File(capture_f5_fname, "a") as f5:
        signal_path = f"read_{read_id}/Signal"
        f5[signal_path] = raw_signal
        f5[signal_path].attrs["read_id"] = read_id
        f5[signal_path].attrs["start_time_bulk"] = metadata.start_time_bulk
        f5[signal_path].attrs["start_time_local"] = metadata.start_time_local
        f5[signal_path].attrs["duration"] = metadata.duration
        f5[signal_path].attrs["ejected"] = metadata.ejected
        f5[signal_path].attrs["voltage"] = metadata.voltage_threshold
        f5[signal_path].attrs["open_channel_pA"] = metadata.open_channel_pA

        channel_path = f"read_{read_id}/channel_id"
        f5.create_group(channel_path)
        f5[channel_path].attrs["channel_number"] = metadata.channel_number
        f5[channel_path].attrs["digitisation"] = metadata.calibration.digitisation
        f5[channel_path].attrs["range"] = metadata.calibration.rng
        f5[channel_path].attrs["offset"] = metadata.calibration.offset
        f5[channel_path].attrs["sampling_rate"] = metadata.sampling_rate
        f5[channel_path].attrs["open_channel_pA"] = metadata.open_channel_pA
