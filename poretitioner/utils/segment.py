import logging
import os

import dask.bag as db
import h5py
import numpy as np
import uuid
from . import raw_signal_utils
from dask.diagnostics import ProgressBar

ProgressBar().register()

__version__ = "0.0.1"  # TODO : Only declare version in one place. Can I get
                       # this from default.nix or somewhere else?
__name__ = "poretitioner"


def apply_capture_filters(capture, filters):
    """
    Check whether a single nanopore capture passes a set of filters. Filters
    are based on summary statistics (e.g., mean) and a range of allowed values.

    Notes on filter behavior: If the filters dict is empty, there are no filters
    and the capture passes. Filters are inclusive of high and low values. Only
    supported filters are allowed. (mean, stdv, median, min, max, length)

    More complex filtering should be done with a custom function.

    Parameters
    ----------
    capture : array or list
        Time series of nanopore current values for a single capture.
    filters : dict
        Keys are strings matching the supported filters, values are a tuple
        giving the endpoints of the valid range. E.g. {"mean": (0.1, 0.5)}
        defines a filter such that 0.1 <= mean(capture) <= 0.5.

    Returns
    -------
    boolean
        True if capture passes all filters; False otherwise.
    """
    logger = logging.getLogger("apply_capture_filters")
    # TODO: Implement logger best practices : https://github.com/uwmisl/poretitioner/issues/12
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    supported_filters = {"mean": np.mean,
                         "stdv": np.std,
                         "median": np.median,
                         "min": np.min,
                         "max": np.max,
                         "length": len}
    pass_filters = True
    for filt, (low, high) in filters.items():
        if filt in supported_filters:
            val = supported_filters[filt](capture)
            if (low is not None and low > val) or \
               (high is not None and val > high):
                pass_filters = False
                return pass_filters
        else:
            # Warn filter not supported
            logger.warning(f"Filter {filt} not supported; ignoring.")
    return pass_filters


def find_captures(signal_pA, signal_threshold_frac, alt_open_channel_pA,
                  terminal_capture_only=False, filters={}, delay=50,
                  end_tol=0):
    """For a single section of current (capture window, in units of pA), identify
    any captures within the window.

    Current is first converted from pA to a fractionalized (0, 1) range based on
    open channel current, and then any spans of current below the threshold
    are considered possible captures. These are further reduced by filters and
    whether the capture must be in the pore at the end of the window.

    Parameters
    ----------
    signal_pA : np.array
        Time series of nanopore current values (in units of pA).
    signal_threshold_frac : float
        Threshold for the first pass of finding captures (in fractional
        current). (Captures are <= the threshold.)
    alt_open_channel_pA : float
        If the open channel current cannot be determined based on the given
        current window (AKA based on signal_pA), use this value instead.
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
    List of tuples
        List of captures in the given window of signal.
    Float
        Open channel current value used to fractionalize raw current.
    """
    # Attempt to find open channel current (if not poss., use alt)
    open_channel_pA = raw_signal_utils.find_open_channel_current(
        signal_pA, alt_open_channel_pA)
    if open_channel_pA is None:
        open_channel_pA = alt_open_channel_pA
    # Convert to frac current
    frac_current = raw_signal_utils.compute_fractional_blockage(
        signal_pA, open_channel_pA)
    del signal_pA
    # Apply signal threshold & get list of captures
    captures = raw_signal_utils.find_segments_below_threshold(
        frac_current, signal_threshold_frac)
    # If last_capture_only, reduce list of captures to only last
    if terminal_capture_only and len(captures) > 1:
        if np.abs(captures[-1][1] - len(frac_current)) <= end_tol:
            captures = [captures[-1]]
        else:
            captures = []

    if delay > 0:
        for i, capture in enumerate(captures):
            capture_start, capture_end = capture
            if capture_end - capture_start > delay:
                capture_start += delay
                captures[i] = (capture_start, capture_end)

    # Apply filters to remaining capture(s)
    filtered_captures = []
    for capture in captures:
        capture_start, capture_end = capture
        if apply_capture_filters(frac_current[capture_start: capture_end], filters):
            filtered_captures.append(capture)
    # Return list of captures
    return filtered_captures, open_channel_pA


def find_captures_dask_wrapper(data_in, terminal_capture_only=False,
                               filters={}, delay=50, end_tol=0):
    """Wrapper for find_captures since dask bag can only take one arg as input,
    plus kwargs. See find_captures() for full description.

    Parameters
    ----------
    data_in : Iterable of signal_pA, signal_threshold_frac, alt_open_channel_pA
        signal_pA : np.array
        Time series of nanopore current values (in units of pA).
        signal_threshold_frac : float
        Threshold for the first pass of finding captures (in fractional
        current). (Captures are <= the threshold.)
        alt_open_channel_pA : float
        If the open channel current cannot be determined based on the given
        current window (AKA based on signal_pA), use this value instead.
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
    List of tuples
        List of captures in the given window of signal.
    Float
        Open channel current value used to fractionalize raw current.
    """
    signal_pA, signal_threshold_frac, alt_open_channel_pA = data_in
    return find_captures(signal_pA, signal_threshold_frac, alt_open_channel_pA,
                         terminal_capture_only=terminal_capture_only,
                         filters=filters, delay=delay, end_tol=end_tol)


def create_capture_fast5(bulk_f5_fname, capture_f5_fname, config,
                         overwrite=False, sub_run=(None, None, None)):
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
    """
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
            g.attrs.create("bulk_filename", bulk_f5_fname,
                           dtype=f"S{len(bulk_f5_fname)}")

            # /Meta/tracking_id
            attrs = ugk["tracking_id"].attrs
            g = capture_f5.create_group("/Meta/tracking_id")
            for k, v in attrs.items():
                g.attrs.create(k, v)

            if sub_run is not None:
                sub_run_id, sub_run_offset, sub_run_duration = sub_run
                if sub_run_id is not None:
                    g.attrs.create("sub_run_id", sub_run_id,
                                   dtype=f"S{len(sub_run_id)}")
                if sub_run_offset is not None:
                    g.attrs.create("sub_run_offset", sub_run_offset)
                if sub_run_duration is not None:
                    g.attrs.create("sub_run_duration", sub_run_duration)

            # /Meta/Segmentation
            # TODO: define config param structure : https://github.com/uwmisl/poretitioner/issues/27
            # config = {"param": "value",
            #           "filters": {"f1": (min, max), "f2: (min, max)"}}
            g = capture_f5.create_group("/Meta/Segmentation")
            print(__name__)
            g.attrs.create("segmenter", __name__, dtype=f"S{len(__name__)}")
            g.attrs.create("segmenter_version", __version__, dtype=f"S{len(__version__)}")
            g_filt = capture_f5.create_group("/Meta/Segmentation/filters")
            for k, v in config.items():
                if k == "filters":
                    for filt, (min_filt, max_filt) in v.items():
                        # Create compound dset for filters
                        dtypes = np.dtype([('min', type(min_filt),
                                           ('max', type(max_filt)))])
                        d = g_filt.create_dataset(k, (2,), dtype=dtypes)
                        d[...] = (min_filt, max_filt)
                else:
                    g.create(k, v)


def _prep_capture_windows(bulk_f5_fname, f5_subsection_start, f5_subsection_end,
                          voltage_t, signal_t, good_channels, open_channel_prior_mean):
    """Helper function to extract raw data from the bulk fast5 for segmentation.

    Parameters
    ----------
    bulk_f5_fname : str
        Filename of the bulk fast5, to be segmented into captures.
    f5_subsection_start : int, optional
        Time point at which to start segmenting, by default None
    f5_subsection_end : int, optional
        Time point at which to stop segmenting, by default None
    voltage_t : int
        Voltage must be at or below this value to be considered a capture window.
    signal_t : float
        Threshold for the first pass of finding captures (in fractional
        current). (Captures are <= the threshold.)
    good_channels : list of ints
        Specifies the nanopore array channels to be segmented.
    open_channel_prior_mean : int
        Approximate estimate of the open channel current value.

    Returns
    -------
    (raw_signals, signal_metadata, run_id, sampling_rate)
        Segments of the raw signal representing capture windows, and metadata
        about these segments (channel number, window endpoints, offset, range,
        digitisation).
    """
    logger = logging.getLogger("_prep_capture_windows")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    with h5py.File(bulk_f5_fname, "r") as bulk_f5:
        logger.info(f"Reading in signals for bulk file: {bulk_f5_fname}")
        voltage = raw_signal_utils.get_voltage(bulk_f5, start=f5_subsection_start, end=f5_subsection_end)
        logger.debug(f"voltage: {voltage}")
        run_id = str(bulk_f5["/UniqueGlobalKey/tracking_id"].attrs.get("run_id"))[2:-1]
        sampling_rate = int(bulk_f5["/UniqueGlobalKey/context_tags"].attrs.get("sample_frequency"))

        logger.info("Identifying capture windows (via voltage threshold).")
        capture_windows = raw_signal_utils.find_segments_below_threshold(
            voltage, voltage_t)
        logger.debug(f"run_id: {run_id}, sampling_rate: {sampling_rate}, "
                     f"#/capture windows: {len(capture_windows)}")
        logger.debug("Prepping raw signals for parallel processing.")
        raw_signals = []  # Input data to find_captures_dask_wrapper
        signal_metadata = []  # Metadata -- no need to pass through the segmenter
        for channel_no in good_channels:
            raw = raw_signal_utils.get_scaled_raw_for_channel(
                bulk_f5, channel_no=channel_no, start=f5_subsection_start,
                end=f5_subsection_end)
            offset, rng, digi = raw_signal_utils.get_scale_metadata(bulk_f5, channel_no)
            for capture_window in capture_windows:
                raw_signals.append((raw[capture_window[0]: capture_window[1]],
                                    signal_t, open_channel_prior_mean))
                signal_metadata.append([channel_no, capture_window, offset, rng, digi])
    return raw_signals, signal_metadata, run_id, sampling_rate


def parallel_find_captures(bulk_f5_fname, config, f5_subsection_start=None, f5_subsection_end=None):
    """Identify captures within the bulk fast5 file in the specified range
    (from f5_subsection_start to f5_subsection_end.)

    If f5_subsection_{start|end} are not specified, the default is to use the
    range of current starting from the beginning and/or end of the bulk fast5.

    Note: config must be validated before this function.

    Parameters
    ----------
    bulk_f5_fname : str
        Filename of the bulk fast5, to be segmented into captures.
    config : dict
        Configuration parameters for the segmenter.
        # TODO : document allowed values : https://github.com/uwmisl/poretitioner/issues/27
    f5_subsection_start : int, optional
        Time point at which to start segmenting, by default None
    f5_subsection_end : int, optional
        Time point at which to stop segmenting, by default None

    Returns
    -------
    list of tuples
        Metadata about the captures. To be used for diagnostic purposes.

    Raises
    ------
    IOError
        If the desired output location does not exist (i.e., save_location) in
        the config, raise IOError.
    """

    logger = logging.getLogger("parallel_find_captures")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    n_workers = config["compute"]["n_workers"]
    assert type(n_workers) is int
    voltage_t = config["segment"]["voltage_threshold"]
    signal_t = config["segment"]["signal_threshold"]
    delay = config["segment"]["translocation_delay"]
    open_channel_prior_mean = config["segment"]["open_channel_prior_mean"]
    good_channels = config["segment"]["good_channels"]
    end_tol = config["segment"]["end_tol"]
    terminal_capture_only = config["segment"]["terminal_capture_only"]
    filters = config["filters"]
    save_location = config["output"]["capture_f5_dir"]  # TODO : Verify exists; don't create (handle earlier)
    n_per_file = config["output"]["captures_per_f5"]
    if f5_subsection_start is None:
        f5_subsection_start = 0

    if not os.path.exists(save_location):
        raise IOError(f"Path to capture file location does not exist: {save_location}")

    raw_signals, context, run_id, sampling_rate = \
        _prep_capture_windows(bulk_f5_fname, f5_subsection_start,
                              f5_subsection_end, voltage_t, signal_t,
                              good_channels, open_channel_prior_mean)

    logger.debug("Loading up the bag with signals.")
    bag = db.from_sequence(raw_signals, npartitions=64)
    capture_map = bag.map(find_captures_dask_wrapper,
                          terminal_capture_only=terminal_capture_only,
                          filters=filters,
                          delay=delay,
                          end_tol=end_tol)
    logger.info("Beginning segmentation.")
    captures = capture_map.compute(num_workers=n_workers)
    assert len(captures) == len(context)
    logger.debug(f"Captures (1st 10): {captures[:10]}")

    # Write captures to fast5
    n_in_file = 0
    file_no = 1
    capture_f5_fname = os.path.join(save_location, f"{run_id}_{file_no}.fast5")
    capture_metadata = []
    for i, ((captures_in_window, open_channel_pA), window_context, (window_raw, _, _)) in \
            enumerate(zip(captures, context, raw_signals)):
        channel_no, capture_window, offset, rng, digi = window_context
        window_start, window_end = capture_window
        for capture in captures_in_window:
            read_id = str(uuid.uuid4())
            start_time_local = capture[0] + window_start  # Start time relative to the start of the bulk f5 subsection
            start_time_bulk = start_time_local + f5_subsection_start  # relative to the start of the entire bulk f5
            capture_duration = capture[1] - capture[0]
            logger.debug(f"Capture duration: {capture_duration}")
            if n_in_file >= n_per_file:
                n_in_file = 0
                file_no += 1
                capture_f5_fname = os.path.join(save_location, f"{run_id}_{file_no}.fast5")
            n_in_file += 1
            raw_pA = window_raw[capture[0]: capture[1]]
            logger.debug(f"Length of raw signal : {len(raw_pA)}")
            write_capture_to_fast5(capture_f5_fname, read_id, raw_pA,
                                   start_time_bulk, start_time_local,
                                   capture_duration, voltage_t, open_channel_pA,
                                   channel_no, digi, offset, rng, sampling_rate)
            capture_metadata.append([capture_f5_fname, read_id,
                                    start_time_bulk, start_time_local,
                                    capture_duration, voltage_t, open_channel_pA,
                                    channel_no, sampling_rate])
    return capture_metadata


def write_capture_to_fast5(capture_f5_fname, read_id, signal_pA, start_time_bulk,
                           start_time_local, duration, voltage, open_channel_pA,
                           channel_no, digitisation, offset, rng, sampling_rate):
    """Write a single capture to the specified capture fast5 file (which has
    already been created via create_capture_fast5()).

    Parameters
    ----------
    capture_f5_fname : str
        Filename of the capture fast5 file to be augmented.\
    read_id : str
        Identifier for this read (unique within the run, usually uuid.)
    signal_pA : np.array
        Time series of nanopore current values (in units of pA).
    start_time_bulk : int
        Starting time of the capture relative to the start of the bulk fast5
        file.
    start_time_local : int
        Starting time of the capture relative to the beginning of the segmented
        region. (Relative to f5_subsection_start in parallel_find_captures().)
    duration : int
        Number of data points in the capture.
    voltage : int
        Voltage at which the capture occurs (single value for entire window).
    open_channel_pA : float
        Median signal value when no capture is in the pore.
    channel_no : int
        Nanopore channel number (in MinION, value is between 1-512).
    digitisation : numeric
        Digitisation value specified in bulk fast5.
    offset : numeric
        Offset value specified in bulk fast5.
    rng : numeric
        Range value specified in bulk fast5.
    sampling_rate : int
        Nanopore sampling rate (Hz)
    """
    path, fname = os.path.split(capture_f5_fname)
    if not os.path.exists(path):
        raise IOError(f"Path to capture file location does not exist: {path}")
    signal_digi = raw_signal_utils.digitize_raw_current(signal_pA, offset, rng,
                                                        digitisation)
    with h5py.File(capture_f5_fname, "a") as f5:
        signal_path = f"read_{read_id}/Signal"
        f5[signal_path] = signal_digi
        f5[signal_path].attrs["read_id"] = read_id
        f5[signal_path].attrs["start_time_bulk"] = start_time_bulk
        f5[signal_path].attrs["start_time_local"] = start_time_local
        f5[signal_path].attrs["duration"] = duration
        f5[signal_path].attrs["voltage"] = voltage
        f5[signal_path].attrs["open_channel_pA"] = open_channel_pA

        channel_path = f"read_{read_id}/channel_id"
        f5.create_group(channel_path)
        f5[channel_path].attrs["channel_number"] = channel_no
        f5[channel_path].attrs["digitisation"] = digitisation
        f5[channel_path].attrs["range"] = rng
        f5[channel_path].attrs["offset"] = offset
        f5[channel_path].attrs["sampling_rate"] = sampling_rate
        f5[channel_path].attrs["open_channel_pA"] = open_channel_pA
