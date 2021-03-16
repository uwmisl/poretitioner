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

from .. import application_info, logger
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
from . import filtering
from .configuration import GeneralConfiguration, SegmentConfiguration
from .core import Window, find_windows_below_threshold

ProgressBar().register()

app_info = application_info.get_application_info()

__version__ = app_info.version
__name__ = app_info.name

__all__ = ["segment"]


def generate_read_id(*args, **kwargs) -> str:
    """Generate a read ID. There's a lot of freedom in how we define these (e.g.
    UUID, a deterministic seed)

    Returns
    -------
    str
        A unique identifier for a read.
    """
    read_id = str(uuid.uuid4())
    return read_id


def find_captures(
    signal_pA: PicoampereSignal,
    channel_number: int,
    capture_window: Window,
    signal_threshold_frac: float,
    open_channel_pA_calculated: float,
    open_channel_pA_prior_range: float = 30,
    terminal_capture_only: bool = False,
    # TODO: Katie Q: DANGER! Mutable obj reference as default
    filters: Optional[List[filtering.FilterPlugin]] = None,
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
    channel_number : int
        Channel this signal_pA originated from.
    signal_threshold_frac : float
        Threshold for the first pass of finding captures (in fractional
        current). (Captures are <= the threshold.)
    open_channel_pA_calculated : float
        Calculated estimate of the open channel current value.
    open_channel_pA_prior_range : float
        When recalculating the local open_channel_pA, accept values up to this
        far away from open_channel_pA_calculated.
    terminal_capture_only : bool, optional
        Only return the final capture in the window, and only if it remains
        captured until the end of the window (to be ejected), by default False
    filters : List[FilterPlugin], optional
        List of filter plugins to apply during segmentation, default None
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

    filters = [] if filters is None else filters

    # Try to locally recalculate open_channel_pA (if not possible, default to input)
    open_channel_pA_calculated = find_open_channel_current(
        signal_pA,
        open_channel_pA_calculated,
        open_channel_pA_prior_range,
        open_channel_pA_calculated,
    )

    # Convert to frac current
    frac_current = signal_pA.to_fractionalized(open_channel_pA_calculated)

    # Apply signal threshold & get list of captures
    potential_captures = find_windows_below_threshold(
        frac_current, signal_threshold_frac
    )
    del frac_current

    captures = [
        Capture(
            signal_pA[potential_capture.start + delay : potential_capture.end],
            channel_number,
            Window(  # Relative to the run, not to the start of the capture window
                potential_capture.start + capture_window.start + delay,
                potential_capture.end + capture_window.start,
            ),
            signal_threshold_frac,
            open_channel_pA_calculated,
            np.abs(capture_window.end - (potential_capture.end + capture_window.start))
            <= end_tol,
        )
        for potential_capture in potential_captures
    ]

    # If terminal_capture_only, reduce list of captures to only last
    if terminal_capture_only:
        if len(captures) >= 1 and captures[-1].ejected:
            captures = [captures[-1]]
        else:
            captures = []

    # Apply filters to remaining capture(s)
    filtered_captures = [
        capture
        for capture in captures
        if filtering.apply_feature_filters(capture.fractionalized, filters)
    ]
    return filtered_captures


def find_captures_dask_wrapper(
    unsegmented_signal: Capture,
    terminal_capture_only=False,
    filters: Optional[List[filtering.FilterPlugin]] = None,
    delay=50,
    end_tol=0,
):
    """Wrapper for find_captures since dask bag can only take one arg as input,
    plus kwargs. See find_captures() for full description.

    Parameters
    ----------
    capture : Capture signal
    terminal_capture_only : bool, optional
        Only return the final capture in the window, and only if it remains
        captured until the end of the window (to be ejected), by default False
    filters : List[FilterPlugin], optional
        List of filter plugins to apply during segmentation, default None
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
    filters = [] if filters is None else filters
    return find_captures(
        unsegmented_signal.signal,  # TODO this is not yet a capture at this point! It's just a region in the time series where there *could* be captures.
        unsegmented_signal.channel_number,
        unsegmented_signal.window,
        unsegmented_signal.signal_threshold_frac,
        unsegmented_signal.open_channel_pA_calculated,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )


@dataclass
class PreppedCaptureCandidates:
    """A series of captures ready to be processed in paralllel.

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
    open_channel_pA_prior: float,
    open_channel_pA_prior_bound: float = 40,
    sub_run_start_seconds: int = 0,
    sub_run_end_seconds: Optional[int] = None,
) -> PreppedCaptureCandidates:
    """Helper function to extract raw data from the bulk fast5 for segmentation.

    Parameters
    ----------
    bulk_f5_fname : str
        Filename of the bulk fast5, to be segmented into captures.
    sub_run_start_seconds : int
        Time point at which to start segmenting, by default 0
    sub_run_end_seconds : int, optional
        Time point at which to stop segmenting, by default None
    voltage_threshold : int
        Voltage must be at or below this value to be considered a capture window.
    signal_threshold_frac : float
        Threshold for the first pass of finding captures (in fractional
        current). (Captures are <= the threshold.)
    good_channels : List[int]
        Specifies the nanopore array channels to be segmented.
    open_channel_pA_prior : float
        Given estimate of the open channel current value (not calculated).
    open_channel_pA_prior_bound : float
        Range of current values allowed on either side of open_channel_pA_prior.

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
        voltage = bulk_f5.get_voltage(
            start=sub_run_start_seconds, end=sub_run_end_seconds
        )
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
        unsegmented_signals = []  # Input data to find_captures_dask_wrapper
        signal_metadata = []  # Metadata -- no need to pass through the segmenter
        for channel_number in good_channels:
            calibration = bulk_f5.get_channel_calibration(channel_number)

            raw_signal: RawSignal = bulk_f5.get_raw_signal(
                channel_number, start=sub_run_start_seconds, end=sub_run_end_seconds
            )
            pico = raw_signal.to_picoamperes()
            open_channel_pA_calculated = find_open_channel_current(
                pico,
                open_channel_guess=open_channel_pA_prior,
                default=open_channel_pA_prior,
                open_channel_bound=open_channel_pA_prior_bound,
            )
            for capture_window in capture_windows:
                start, end = capture_window.start, capture_window.end

                unsegmented_signal = Capture(
                    pico[start:end],
                    channel_number,
                    capture_window,
                    signal_threshold_frac,
                    open_channel_pA_calculated,
                    None,
                )
                metadata = SignalMetadata(
                    channel_number=channel_number,
                    window=capture_window,
                    calibration=calibration,
                )

                unsegmented_signals.append(unsegmented_signal)
                signal_metadata.append(metadata)
        prepped = PreppedCaptureCandidates(
            unsegmented_signals, signal_metadata, run_id, sampling_rate
        )
        return prepped


def segment(
    bulk_f5_filepath,
    config: GeneralConfiguration,
    segment_config: SegmentConfiguration,
    save_location=None,
    filters: Optional[List[filtering.FilterPlugin]] = None,
    overwrite=True,
    sub_run_start_seconds=0,
    sub_run_end_seconds=None,
) -> List[CaptureMetadata]:
    """Identifies the capture regions in a nanopore ionic current signal.

    Parameters
    ----------
    bulk_f5_filepath : str
        Path to the bulk fast5 file, likely generated by the MinKnow software.
    save_location : str
        Directory to save the segmentation results (results will often be saved to more than one file).
    config : GeneralConfiguration
        General run configuration.
    segment_config : SegmentConfiguration
        Segmentation configuration.
    sub_run_start_seconds : int, optional
        Where to start in the run (e.g. start segmenting after 10 seconds). This is useful in cases
        where the run is continuous, but you added a different analyte or wash at some known point in time, by default 0
    sub_run_end_seconds : int, optional
        Where to stop segmenting in the run (if anywhere). This is useful in cases
        where the run is continuous, but you added a different analyte or wash at some known point in time, by default None
    """
    filters = [] if filters is None else filters
    local_logger = logger.getLogger()

    local_logger.debug(
        f"Segmenting bulk file '{bulk_f5_filepath}' and saving results at location '{save_location}' "
    )
    local_logger.debug(
        f"Segmenting configuration '{bulk_f5_filepath}' and saving results at location '{save_location}' "
    )
    return parallel_find_captures(
        config,
        segment_config,
        filters=filters,
        save_location=save_location,
        overwrite=overwrite,
        sub_run_start_seconds=sub_run_start_seconds,
        sub_run_end_seconds=sub_run_end_seconds,
    )


def parallel_find_captures(
    config: GeneralConfiguration,
    segment_config: SegmentConfiguration,
    # Only supporting range filters for now (MVP). This should be expanded to all FilterPlugins: https://github.com/uwmisl/poretitioner/issues/67
    filters: List[filtering.FilterPlugin] = None,
    save_location=None,
    overwrite=False,
    sub_run_start_seconds=0,
    sub_run_end_seconds=None,
) -> List[CaptureMetadata]:
    """Identify captures within the bulk fast5 file in the specified range
    (from sub_run_start_seconds to sub_run_end_seconds.)

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
    sub_run_start_seconds : int, optional
        Time point at which to start segmenting, by default None
    sub_run_end_seconds : int, optional
        Time point at which to stop segmenting, by default None
    overwrite : bool, optional
        If segmented files already exist, overwrite them if True, otherwise.

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
    bulk_f5_fname = segment_config.bulkfast5
    local_logger = logger.getLogger()
    local_logger.setLevel(0)
    # TODO: Update with actual configuring https://github.com/uwmisl/poretitioner/issues/73
    n_workers = config.n_workers
    assert type(n_workers) is int
    voltage_threshold = segment_config.voltage_threshold
    signal_threshold_frac = segment_config.signal_threshold_frac
    delay = segment_config.translocation_delay
    open_channel_prior_mean = segment_config.open_channel_prior_mean
    good_channels = segment_config.good_channels
    end_tol = segment_config.end_tolerance
    terminal_capture_only = segment_config.terminal_capture_only
    save_location = save_location if save_location else config.capture_directory
    n_per_file = segment_config.n_captures_per_file
    filters = [] if filters is None else filters

    if not os.path.exists(save_location):
        raise IOError(f"Path to capture file location does not exist: {save_location}")

    prepped_captures: PreppedCaptureCandidates = _prep_capture_windows(
        bulk_f5_fname,
        voltage_threshold,
        signal_threshold_frac,
        good_channels,
        open_channel_prior_mean,
        sub_run_start_seconds=sub_run_start_seconds,
        sub_run_end_seconds=sub_run_end_seconds,
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
    capture_metadatum = []

    def get_capture_metadata(
        read_id: str,
        sampling_rate: int,
        voltage_threshold: int,
        capture: Capture,
        metadata: SignalMetadata,
        sub_run_start_seconds: int = 0,
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
            start_time_local + sub_run_start_seconds
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

    # with BulkFile(bulk_f5_fname, "r") as bulk:
    #     for captures, metadata in zip(captures, context):
    #         for cap in captures:
    #             # Create a read ID, if this capture's signal doesn't already have one.
    #             read_id = cap.signal.read_id
    #             read_id = read_id if read_id is not None else generate_read_id()
    #             print(read_id)
    #             start, end = cap.window.start, cap.window.end
    #             capture_metadata = get_capture_metadata(
    #                 read_id,
    #                 sampling_rate,
    #                 voltage_threshold,
    #                 cap,
    #                 metadata,
    #                 sub_run_start_seconds=sub_run_start_seconds,
    #             )

    #             capture_metadatum.append(capture_metadata)

    #             if n_in_file >= n_per_file:
    #                 n_in_file = 0
    #                 file_no += 1

    #             capture_f5_filepath = Path(save_location, f"{run_id}_{file_no}.fast5")
    #             print(f"capture_fname:{capture_f5_filepath} n_in_file:{n_in_file}")

    #             # Set writing mode (detailing cases explicitly here)
    #             mode = ""
    #             if (
    #                 n_in_file == 0
    #                 and os.path.exists(capture_f5_filepath)
    #                 and not overwrite
    #             ):
    #                 # CASE:
    #                 #   * We're on the first read to write for this file
    #                 #   * The file already exists
    #                 #   * We're not supposed to overwrite this file
    #                 # RESULT: IOError
    #                 # TODO Verify that the folder is empty waaay up top, not here.
    #                 raise IOError(
    #                     f"File ({capture_f5_filepath}) already exists and overwrite set to False."
    #                 )
    #             elif n_in_file == 0 and overwrite:
    #                 # CASE:
    #                 #   * We're on the first read to write for this file
    #                 #   * Overwrite is set
    #                 # RESULT: Create a new file, even if it means overwriting an existing file
    #                 # TODO : This can actually create a bug, example scenario: what if we have overwrite=True and
    #                 # this run only generates 3 files, but the previous generated 5? The last 2 files would still
    #                 # be there. Should just delete everything in the save_location before writing
    #                 mode = "w"  # Create new file, truncate if exists
    #             else:
    #                 # CASE:
    #                 #   * We're on the nth read to write for this file
    #                 #   * We already handled whether to open a new one, so just write to this file.
    #                 # RESULT: Write to the existing file.
    #                 mode = "r+"  # Read/write, file must exist

    #             # TODO: The current loop constanly re-opens the capture_file! Replace with loop that keeps a
    #             # file open until it fills up.

    #             n_in_file += 1

    #             picoamperes: PicoampereSignal = cap.signal
    #             raw_signal = picoamperes.to_raw()[start:end]
    #             local_logger.debug(f"\tLength of raw signal : {len(raw_signal)}")

    #             local_logger.debug(f"\tWriting to file: {capture_f5_filepath}...")

    #             # mode = "w" if overwrite else "r+"
    #             print(mode)
    #             with CaptureFile(capture_f5_filepath, mode=mode) as capture_file:
    #                 print("#/reads in file before writing", len(capture_file.reads))
    #                 capture_file.initialize_from_bulk(
    #                     bulk, filters, segment_config, sub_run=None
    #                 )
    #                 print("HERE")
    #                 capture_file.write_capture(raw_signal, capture_metadata)
    #                 print("#/reads in file after writing", len(capture_file.reads))

    #             local_logger.debug(f"\tWritten!")

    # Delete existing files if they already exist
    if overwrite and len(os.listdir(save_location)) > 0:
        for f in os.listdir(save_location):
            if f.endswith("fast5"):
                os.unlink(os.path.join(save_location, f))

    with BulkFile(bulk_f5_fname, "r") as bulk:
        # Create first capture_file
        capture_f5_filepath = Path(save_location, f"{run_id}_{file_no}.fast5")
        local_logger.info(f"capture_fname:{capture_f5_filepath} n_in_file:{n_in_file}")
        capture_file = CaptureFile(capture_f5_filepath, mode="w")
        capture_file.initialize_from_bulk(bulk, filters, segment_config, sub_run=None)

        for captures, metadata in zip(captures, context):
            for cap in captures:
                # Create a read ID, if this capture's signal doesn't already have one.
                read_id = cap.signal.read_id
                read_id = read_id if read_id is not None else generate_read_id()
                print(read_id)
                start, end = cap.window.start, cap.window.end
                capture_metadata = get_capture_metadata(
                    read_id,
                    sampling_rate,
                    voltage_threshold,
                    cap,
                    metadata,
                    sub_run_start_seconds=sub_run_start_seconds,
                )

                capture_metadatum.append(capture_metadata)

                if n_in_file >= n_per_file:
                    # File full! Close the old and open a new one!
                    n_in_file = 0
                    file_no += 1
                    capture_file.f5.close()
                    capture_f5_filepath = Path(
                        save_location, f"{run_id}_{file_no}.fast5"
                    )
                    print(f"capture_fname:{capture_f5_filepath} n_in_file:{n_in_file}")
                    mode = "w"
                    capture_file = CaptureFile(capture_f5_filepath, mode=mode)
                    capture_file.initialize_from_bulk(
                        bulk, filters, segment_config, sub_run=None
                    )

                n_in_file += 1

                picoamperes: PicoampereSignal = cap.signal
                raw_signal = picoamperes.to_raw()[start:end]
                local_logger.debug(f"\tLength of raw signal : {len(raw_signal)}")

                local_logger.debug(f"\tWriting to file: {capture_f5_filepath}...")

                # with CaptureFile(capture_f5_filepath, mode=mode) as capture_file:
                print("#/reads in file before writing", len(capture_file.reads))
                capture_file.write_capture(raw_signal, capture_metadata)
                print("#/reads in file after writing", len(capture_file.reads))

                local_logger.debug(f"\tWritten!")

    return capture_metadatum
