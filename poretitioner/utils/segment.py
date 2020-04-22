import logging
import os

import dask.bag as db
import h5py
import numpy as np
import pandas as pd
from . import raw_signal_utils
from dask.diagnostics import ProgressBar

ProgressBar().register()

__version__ = "0.0.1"  # TODO : Only declare version in one place. Can I get
                       # this from default.nix or somewhere else?
__name__ = "poretitioner"


def apply_capture_filters(capture, filters):
    # TODO : Write fn to apply all filters to single capture.
    # If no filters, it should pass
    # Only allow supported filters
    # Filters are inclusive of low & high values
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
            if low > val or val > high:
                pass_filters = False
                return pass_filters
        else:
            # Warn filter not supported
            logger.warning(f"Filter {filt} not supported; ignoring.")
    return pass_filters


def find_segments_below_threshold(time_series, threshold):
    """find_segments_below_threshold

    Find regions where the time series data points drop at or below the
    specified threshold.

    Parameters
    ----------
    time_series : np.array
        Array containing time series data points.
    threshold : numeric
        Find regions where the time series drops at or below this value

    Returns
    -------
    zip iterator
        Each item in the iterator represents the (start, end) points of regions
        where the input array drops at or below the threshold.
    """
    diff_points = np.where(np.abs(np.diff(
        np.where(time_series <= threshold, 1, 0))) == 1)[0]
    if time_series[0] <= threshold:
        diff_points = np.hstack([[0], diff_points])
    if time_series[-1] <= threshold:
        diff_points = np.hstack([diff_points, [len(time_series)]])
    return zip(diff_points[::2], diff_points[1::2])


def find_captures(signal_pA, signal_threshold_frac, alt_open_channel_pA,
                  last_capture_only=False, filters={}, delay=50):
    """[summary]

    Parameters
    ----------
    signal_pA : np.array
        Time series of nanopore current values (in units of pA).
    signal_threshold_frac : float
        Threshold for the first pass of finding captures (in ). (Captures are <= the
        threshold.)
    alt_open_channel_pA : [type]
        [description]
    last_capture_only : bool, optional
        [description], by default False
    filters : dict, optional
        [description], by default {}
    delay : int, optional
        Translocation delay. At high sampling rates, there may be some data
        points at the beginning of the capture that represent the initial
        translocation of the molecule. The beginning of the capture will be
        trimmed by _delay_ points, by default 50.

    Returns
    -------
    List of tuples
        List of captures in the given window of signal.
    """
    # For a single capture window (given as signal_pA):
    # Attempt to find open channel current (if not poss., use alt)
    open_channel_pA = raw_signal_utils.find_open_channel_current(
        signal_pA, alt_open_channel_pA)
    if open_channel_pA is None:
        open_channel_pA = alt_open_channel_pA
    # Convert to frac current
    frac_current = raw_signal_utils.compute_fractional_blockage(
        signal_pA, open_channel_pA)
    # Apply signal threshold & get list of captures
    captures = find_segments_below_threshold(frac_current,
                                             signal_threshold_frac)
    # If last_capture_only, reduce list of captures to only last
    if last_capture_only and len(captures) > 1:
        captures = captures[-1]
    # Apply filters to remaining capture(s)
    filtered_captures = []
    for capture in captures:
        if len(capture) > delay:
            capture = capture[delay:]
        else:
            continue
        if apply_capture_filters(capture, filters):
            filtered_captures.append(capture)
    # Return list of captures
    return filtered_captures


def find_peptides(signal, voltage, signal_threshold=0.7, voltage_threshold=-180):
    # TODO deprecate
    diff_points = np.where(
        np.abs(
            np.diff(
                np.where(
                    np.logical_and(voltage <= voltage_threshold, signal <= signal_threshold), 1, 0
                )
            )
        )
        == 1
    )[0]
    if voltage[0] <= voltage_threshold and signal[0] <= signal_threshold:
        diff_points = np.hstack([[0], diff_points])
    if voltage[-1] <= voltage_threshold and signal[-1] <= signal_threshold:
        diff_points = np.hstack([diff_points, [len(voltage)]])

    return zip(diff_points[::2], diff_points[1::2])


def _find_peptides_helper(raw_signal_meta, voltage=None,
                          open_channel_prior=220,
                          open_channel_prior_stdv=35, signal_threshold=0.7,
                          voltage_threshold=-180, min_duration_obs=0,
                          voltage_change_delay=3):
    """Identify blockages in a single channel of raw current.

    Parameters
    ----------
    raw_signal_meta : [type]
        [description]
    voltage : [type], optional
        [description], by default None
    open_channel_prior : int, optional
        [description], by default 220
    open_channel_prior_stdv : int, optional
        [description], by default 35
    signal_threshold : float, optional
        [description], by default 0.7
    voltage_threshold : int, optional
        [description], by default -180
    min_duration_obs : int, optional
        [description], by default 0
    voltage_change_delay : int, optional
        [description], by default 3

    Returns
    -------
    list of tuples
        Metadata about captures found within the raw signal. Includes the run
        name, channel number, start & end, duration, mean/stdv/median/min/max
        of the fractional signal and the open channel current (in pA).
    """
    run, channel, raw_signal = raw_signal_meta
    peptide_metadata = []
    open_channel = raw_signal_utils.find_open_channel_current(
        raw_signal, open_channel_guess=open_channel_prior, bound=open_channel_prior_stdv
    )
    if open_channel is None:
        open_channel = open_channel_prior
    frac_signal = raw_signal_utils.compute_fractional_blockage(raw_signal, open_channel)
    peptide_segments = find_peptides(frac_signal, voltage,
                                     signal_threshold=signal_threshold,
                                     voltage_threshold=voltage_threshold)
    for peptide_segment in peptide_segments:
        if peptide_segment[1] - peptide_segment[0] - voltage_change_delay < min_duration_obs:
            continue
        peptide_start = peptide_segment[0] + voltage_change_delay
        peptide_end = peptide_segment[1]
        peptide_signal = frac_signal[peptide_start:peptide_end]
        peptide_metadata.append(
            (
                run,
                channel,
                peptide_start,
                peptide_end,
                peptide_end - peptide_start,
                np.mean(peptide_signal),
                np.std(peptide_signal),
                np.median(peptide_signal),
                np.min(peptide_signal),
                np.max(peptide_signal),
                open_channel,
            )
        )
    return peptide_metadata


def create_capture_fast5(bulk_f5_fname, capture_f5_fname, config,
                         overwrite=False, sub_run=(None, None, None)):
    """Prepare a fast5 file to contain the results of segmentation (capture
    fast5 file).

    Parameters
    ----------
    bulk_f5_fname : str
        Filename of the bulk fast5 that will eventually be segmented into
        captures.
        # TODO not sure if this should be a fname or open h5py obj
        The implication is for parallelization, competing for the fh.
    capture_f5_fname : str
        Filename of the capture fast5 file to be created/written. Directory
        containing this file must already exist.
    config : dict
        Configuration parameters for the segmenter.
        # TODO document allowed values
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

    with h5py.File(capture_f5_fname, "w") as capture_f5:
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


def parallel_find_peptides(
    f5_fnames,
    good_channel_dict,
    open_channel_prior,
    open_channel_prior_stdv,
    signal_threshold,
    voltage_threshold,
    min_duration_obs,
    save_location=".",
    save_prefix="segmented_peptides",
    voltage_change_delay=3,
    n_workers=1,
):
    logger = logging.getLogger("parallel_find_peptides")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    for run, f5_fname in f5_fnames.iteritems():
        logger.info("Reading in signals for run: %s" % run)
        f5 = h5py.File(f5_fname)
        voltage = f5.get("/Device/MetaData").value["bias_voltage"] * 5.0
        good_channels = good_channel_dict.get(run)
        raw_signals = []

        for channel_no in good_channels:
            channel = "Channel_%s" % str(channel_no)
            logger.debug(channel)
            raw_signal = raw_signal_utils.get_scaled_raw_for_channel(f5, channel=channel)
            raw_signals.append((run, channel, raw_signal))

        logger.debug("Loading up the bag with signals.")
        bag = db.from_sequence(raw_signals, npartitions=128)
        peptide_map = bag.map(
            _find_peptides_helper,
            voltage=voltage,
            open_channel_prior=open_channel_prior,
            open_channel_prior_stdv=open_channel_prior_stdv,
            signal_threshold=signal_threshold,
            voltage_threshold=voltage_threshold,
            min_duration_obs=min_duration_obs,
            voltage_change_delay=voltage_change_delay,
        )
        logger.debug("Running peptide segmenter.")
        peptide_metadata_by_channel = peptide_map.compute(num_workers=n_workers)
        logger.debug("Converting list of peptides to a dataframe.")
        peptide_metadata = []
        while len(peptide_metadata_by_channel) > 0:
            peptide_metadata.extend(peptide_metadata_by_channel.pop())
        peptide_metadata_df = pd.DataFrame.from_records(
            peptide_metadata,
            columns=[
                "run",
                "channel",
                "start_obs",
                "end_obs",
                "duration_obs",
                "mean",
                "stdv",
                "median",
                "min",
                "max",
                "open_channel",
            ],
        )
        save_name = save_prefix + "_%s.pkl" % (run)
        try:
            os.makedirs(save_location)
        except OSError:
            pass
        logger.debug("Saving dataframe to pickle.")
        peptide_metadata_df.to_pickle(os.path.join(save_location, save_name))


def extract_raw_data(
    f5_fnames,
    df_location=".",
    df_prefix="segmented_peptides",
    save_location=".",
    save_prefix="segmented_peptides_raw_data",
    open_channel_prior=220.0,
    open_channel_prior_stdv=35.0,
):
    logger = logging.getLogger("extract_raw_data")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    for run, f5_fname in f5_fnames.iteritems():
        logger.info("Saving data from %s" % run)
        df = np.load(os.path.join(df_location, df_prefix + "_%s.pkl" % run), allow_pickle=True)
        f5 = h5py.File(f5_fname, "r")
        peptides = []
        last_channel = None
        for i, row in df.iterrows():
            if row.channel != last_channel:
                last_channel = row.channel
                raw_signal = raw_signal_utils.get_scaled_raw_for_channel(f5, channel=row.channel)
                if "open_channel" in row.index:
                    logger.debug("Attempting to get open channel from peptide " "df.")
                    open_channel = row.open_channel
                else:
                    logger.debug("Attempting to find open channel current.")
                    open_channel = raw_signal_utils.find_open_channel_current(
                        raw_signal, open_channel_guess=open_channel_prior, bound=open_channel_prior_stdv
                    )
                    if open_channel is None:
                        logger.debug("Open channel couldn't be found, using " "the given prior.")
                        open_channel = open_channel_prior
                open_channel = np.floor(open_channel)

                logger.debug("Computing fractional current.")
                frac_signal = raw_signal_utils.compute_fractional_blockage(raw_signal, open_channel)

            peptide_signal = frac_signal[row["start_obs"]: row["end_obs"]]
            logger.debug(
                "Mean in df: %0.4f, \tMean in extracted: %0.4f"
                % (row["mean"], np.mean(peptide_signal))
            )
            logger.debug(
                "Len in df: %d, \tLen in extracted: %d"
                % (row["duration_obs"], len(peptide_signal))
            )
            peptides.append(peptide_signal)
        logger.debug("Saving to file.")
        assert len(df) == len(peptides)
        np.save(os.path.join(save_location, save_prefix + "_%s.npy" % run), peptides)
