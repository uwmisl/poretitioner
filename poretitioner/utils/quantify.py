"""
===========
quantify.py
===========


This module contains functionality for quantifying nanopore captures.

"""
import re

import h5py
import numpy as np

from poretitioner.utils import classify, raw_signal_utils


def quantify_files(
    config,
    fast5_fnames,
    filter_name=None,
    quant_method="time_between_captures",
    classified_only=False,
    interval_mins=None,
):
    """Quantify how often captures occur in a batch of FAST5 files.

    This works best with a lenient filtering at the start to identify all blockages.
    Use unfiltered captures as "blockages" and filtered captures as "captures".

    Parameters
    ----------
    config : dict
        Configuration parameters for the segmenter.
        # TODO : document allowed values : https://github.com/uwmisl/poretitioner/issues/27
    fast5_fnames : list of str
        FAST5 filenames containing the reads to classify
    filter_name : str, optional
        If only some reads should be considered as captures, specify a filter that
        contains only those reads. Specify None to use all reads, by default None
    quant_method : str, optional
        The quantification method to run, by default "time_between_captures"
    classified_only : bool, optional
        Only quantify captures that have been successfully classified, by default False
    interval_mins : int, optional
        When quantifying captures over time, this is the length of the window of
        time considered at once. Specify None to quantify the entire file, by default None

    Returns
    -------
    list of float
        Quantification results where each data point represents one window of time.
        If one value is returned in this list, it represents the entire file.

    Raises
    ------
    ValueError
        Raised if the quantification method (quant_method) is not supported.
    """

    # This also requires having a lenient filtering at the start to identify all blockages.
    # Use unfiltered captures as "blockages" and filtered captures as "captures".

    if filter_name:
        read_path = f"/Filter/{filter_name}/pass"
    else:
        read_path = "/"
    blockage_path = "/"

    if "classify" in config:
        classification_path = f"/Classification/{config['classify']['classifier']}"
    else:
        classification_path = None

    capture_arrays = []
    blockage_arrays = []

    if len(fast5_fnames) == 0:
        raise ValueError("No fast5 files specified (len(fast5_fnames) == 0).")

    # The challenge here is that reads will be in different files. Need to read
    # in all the reads from all the files, sort them by channel and time, and
    # then quantify.
    for fast5_fname in fast5_fnames:
        with h5py.File(fast5_fname, "r") as f5:
            sampling_rate = raw_signal_utils.get_sampling_rate(f5)
            capture_array = get_capture_details_in_f5(f5, read_path, classification_path)
            if len(capture_array) > 0:
                if classified_only:
                    ix = np.where(capture_array[:, 4] != "-1")[0]
                    capture_array = capture_array[ix, :]
                capture_arrays.append(capture_array)
            # Get all of the "blockages" -- all regions where the pore is
            # assumed to be blocked by something (capture or not)
            if read_path == blockage_path:
                blockage_array = capture_array
            else:
                blockage_array = get_capture_details_in_f5(f5, read_path, classification_path)
            if len(blockage_array) > 0:
                blockage_arrays.append(blockage_array)

    captures_by_channel = sort_captures_by_channel(np.vstack(capture_arrays))
    blockages_by_channel = sort_captures_by_channel(np.vstack(blockage_arrays))

    del capture_arrays, blockage_arrays

    # The capture windows are stored identically in each read fast5, so no need for loop
    capture_windows = get_capture_windows_by_channel(fast5_fname)
    first_channel = list(capture_windows.keys())[0]

    start_obs = capture_windows[first_channel][0][0]  # Start of first window
    end_obs = capture_windows[first_channel][-1][1]  # End of last window
    # Intervals in format [(start time, end time), ...]
    if interval_mins is None:
        intervals = [(start_obs, end_obs)]
    else:
        interval_obs = interval_mins * 60 * sampling_rate
        intervals = list(range(interval_obs + start_obs, end_obs + 1, interval_obs))
        intervals = list(zip([start_obs] + intervals, intervals + [end_obs]))

    if quant_method == "time_between_captures":
        capture_times = calc_time_until_capture_intervals(
            capture_windows,
            captures_by_channel,
            blockages_by_channel=blockages_by_channel,
            intervals=intervals,
        )
        return capture_times

    elif quant_method == "capture_freq":
        capture_freqs = calc_capture_freq_intervals(
            capture_windows,
            captures_by_channel,
            intervals=intervals,
            sampling_rate=sampling_rate,
        )
        return capture_freqs

    else:
        raise ValueError(f"Quantification method not implemented: {quant_method}")


def get_capture_windows_by_channel(fast5_fname):
    """Extract capture windows (regions where the voltage is appropriate to accept
    a captured peptide or other analyte) from a fast5 file, in a dictionary and
    sorted by channel

    Parameters
    ----------
    fast5_fname : string
        Location of a fast5 file.

    Returns
    -------
    dict
        {channel_no: [(window_start, window_end), ...]}
    """
    base_path = "/Meta/Segmentation/capture_windows"
    windows_by_channel = {}
    with h5py.File(fast5_fname, "r") as f5:
        for grp_name in f5.get(base_path):
            if "Channel" in grp_name:
                channel_no = int(re.findall(r"Channel_(\d+)", grp_name)[0])
                windows = f5.get(f"{base_path}/{grp_name}")[()]
                windows_by_channel[channel_no] = windows
    return windows_by_channel


def sort_captures_by_channel(capture_array):
    """Take a list of captures (defined below), and sort it to produce a dictionary
    of {channel: [capture0, capture1, ...]}, where the captures are sorted in order.

    The format of each capture is [read_id, capture_start, capture_end, channel_no, assigned_class].

    Parameters
    ----------
    capture_array : list
        list of lists: [[read_id, capture_start, capture_end, channel_no, assigned_class]]

    Returns
    -------
    dict
        {channel_no: [(capture_start, capture_end), ...]}
    """
    # Initialize dictionary of {channel: [captures]}
    channels = np.unique(capture_array[:, 3])
    captures_by_channel = {}
    for channel in channels:
        captures_by_channel[int(channel)] = []

    # Populate dictionary values
    for row in capture_array:
        channel = int(row[3])
        captures_by_channel[channel].append(row)

    # Sort the captures within each channel by time
    for channel, captures in captures_by_channel.items():
        captures = np.array(captures)
        captures = captures[captures[:, 1].argsort()]
        capture_regions = []
        for _, capture_start, capture_end, _, _ in captures:
            capture_regions.append((int(capture_start), int(capture_end)))
        captures_by_channel[channel] = capture_regions

    return captures_by_channel


def get_capture_details_in_f5(f5, read_path=None, classification_path=None):
    """Get capture times & channel info for all reads in the fast5 at read_path.
    If a classification_path is specified, get the label for each read too.

    Parameters
    ----------
    f5 : h5py.File
        Open fast5 file in read mode.
    read_path : str, optional
        Location of the reads to retrieve, by default None (if None, defaults
        to "/" for the read location.)
    classification_path : str, optional
        Location of the classification results, by default None (if None, no
        labels are returned)

    Returns
    -------
    np.array
        Array containing read_id, capture/read start & end, channel number, and
        the classification label (if applicable). Shape is (5 x n_reads).
    """

    read_path = read_path if read_path is not None else "/"
    read_h5group_names = f5.get(read_path)

    captures_in_f5 = []

    if read_h5group_names is None:
        return None

    for grp_name in read_h5group_names:
        if "read" not in grp_name:
            continue
        read_id = re.findall(r"read_(.*)", grp_name)[0]
        grp = f5.get(grp_name)
        a = grp["Signal"].attrs
        capture_start = a["start_time_local"]
        capture_end = capture_start + a["duration"]

        a = grp["channel_id"].attrs
        channel_no = int(a["channel_number"])

        if classification_path:
            (
                pred_class,
                prob,
                assigned_class,
                passed_classification,
            ) = classify.get_classification_for_read(f5, read_id, classification_path)
        else:
            assigned_class = None

        captures_in_f5.append((read_id, capture_start, capture_end, channel_no, assigned_class))

    return np.array(captures_in_f5)


def calc_time_until_capture(capture_windows, captures, blockages=None):
    """calc_time_until_capture

    Finds all times between captures from a single channel. This is defined
    as the open pore time from the end of the previous capture to the
    current capture. Includes subtracting other non-capture blockages since
    those blockages reduce the amount of overall open pore time.

    Parameters
    ----------
    capture_windows : list of tuples of ints [(start, end), ...]
        Regions of current where the nanopore is available to accept a
        capture. (I.e., is in a "normal" voltage state.) [(start, end), ...]
    captures : list of tuples of ints [(start, end), ...]
        Regions of current where a capture is residing in the pore. The
        function is calculating time between these values (minus blockages).
    blockages : list of tuples of ints [(start, end), ...]
        Regions of current where the pore is blocked by any capture or non-
        capture. These are removed from the time between captures, if
        specified.

    Returns
    -------
    list of floats
        List of all capture times from a single channel.
    """
    # TODO: Implement logger best practices : https://github.com/uwmisl/poretitioner/issues/12
    # local_logger = logger.getLogger()

    all_capture_times = []

    if captures is None or len(captures) == 0:
        return None

    elapsed_time_until_capture = 0
    for capture_window_i, capture_window in enumerate(capture_windows):
        # Get all the captures & blockages within that window
        captures_in_window = raw_signal_utils.get_overlapping_regions(capture_window, captures)
        if blockages is not None:
            blockages_in_window = raw_signal_utils.get_overlapping_regions(
                capture_window, blockages
            )
        else:
            blockages_in_window = []
        # If there are no captures in the window, add the window to the elapsed
        # time and subtract any blockages.
        if len(captures_in_window) == 0:
            elapsed_time_until_capture += capture_window[1] - capture_window[0]
            for blockage in blockages_in_window:
                elapsed_time_until_capture -= blockage[1] - blockage[0]
            continue
        # If there's a capture in the window, add the partial window to the
        # elapsed time. Subtract blockages that came before the capture.
        else:
            last_capture_end = capture_window[0]
            for capture_i, capture in enumerate(captures_in_window):
                elapsed_time_until_capture += capture[0] - last_capture_end
                for blockage in blockages_in_window:
                    # Blockage must start after the last capture ended and
                    # finish before the next capture starts; otherwise skip
                    if blockage[0] >= last_capture_end and blockage[1] < capture[0]:
                        elapsed_time_until_capture -= blockage[1] - blockage[0]
                        blockages.pop(0)
                all_capture_times.append(elapsed_time_until_capture)
                # Reset elapsed time.
                elapsed_time_until_capture = max(capture_window[1] - capture[1], 0)
                captures.pop(0)
                last_capture_end = capture[1]
    return all_capture_times


def calc_time_until_capture_intervals(
    capture_windows, captures_by_channel, blockages_by_channel=None, intervals=[]
):
    """Compute time_until_capture across specified intervals of time.

    Parameters
    ----------
    capture_windows : list of tuples of ints [(start, end), ...]
        Regions of current where the nanopore is available to accept a
        capture. (I.e., is in a "normal" voltage state.) [(start, end), ...]
    captures_by_channel : dictionary of captures {channel_no: [(start, end), ...]}
        Regions of current where a capture is residing in the pore. The
        function is calculating time between these values (minus blockages).
    blockages_by_channel : dictionary of blockages {channel_no: [(start, end), ...]}
        Regions of current where the pore is blocked by any capture or non-
        capture. These are removed from the time between captures, if
        specified.
    intervals : list, optional
        List of start and end times, in observations (#/datapoints, not minutes), by default []

    Returns
    -------
    list of floats
        List of the average time between captures, where each item in this list
        corresponds to an entry in the input list intervals.
    """
    assert len(intervals) > 0
    capture_times = []
    elapsed_time_obs = 0
    for interval in intervals:
        start_obs, end_obs = interval
        # Compute capture freq for all channels separately, pooling results
        interval_capture_times = []
        for channel in captures_by_channel.keys():
            captures = captures_by_channel[channel]
            captures = raw_signal_utils.get_overlapping_regions(interval, captures)
            blockages = blockages_by_channel[channel]
            blockages = raw_signal_utils.get_overlapping_regions(interval, blockages)
            windows = capture_windows[channel]
            windows = raw_signal_utils.get_overlapping_regions(interval, windows)

            assert captures is not None
            assert blockages is not None
            assert windows is not None
            ts = calc_time_until_capture(windows, captures, blockages=blockages)
            if ts is None:
                elapsed_time_obs += end_obs - start_obs
            else:
                ts = [ts[0] + elapsed_time_obs] + ts[1:]
                interval_capture_times.extend(ts)
                elapsed_time_obs = 0
        capture_times.append(np.mean(interval_capture_times))
    return capture_times


def calc_capture_freq(capture_windows, captures):
    """Calculate the capture frequency -- the number of captures per unit of time.

    Parameters
    ----------
    capture_windows : list of tuples of ints [(start, end), ...]
        Regions of current where the nanopore is available to accept a
        capture. (I.e., is in a "normal" voltage state.) [(start, end), ...]
    captures : list of tuples of ints [(start, end), ...]
        Regions of current where a capture is residing in the pore. The
        function is calculating time between these values (minus blockages).

    Returns
    -------
    list of floats
        List of all capture times from a single channel.
    """

    if captures is None:
        return None

    if len(captures) == 0:
        return 0

    n_captures = 0

    for capture_window_i, capture_window in enumerate(capture_windows):
        window_start, window_end = capture_window

        # Get all the captures within that window
        captures_in_window = raw_signal_utils.get_overlapping_regions(capture_window, captures)
        n_captures += len(captures_in_window)
    return n_captures


def calc_capture_freq_intervals(
    capture_windows,
    captures_by_channel,
    intervals=[],
    sampling_rate=10000,
):
    """Compute capture_freq across specified intervals of time.

    Parameters
    ----------
    capture_windows : list of tuples of ints [(start, end), ...]
        Regions of current where the nanopore is available to accept a
        capture. (I.e., is in a "normal" voltage state.) [(start, end), ...]
    captures_by_channel : dictionary of captures {channel_no: [(start, end), ...]}
        Regions of current where a capture is residing in the pore. The
        function is calculating time between these values (minus blockages).
    blockages_by_channel : dictionary of blockages {channel_no: [(start, end), ...]}
        Regions of current where the pore is blocked by any capture or non-
        capture. These are removed from the time between captures, if
        specified.
    intervals : list, optional
        List of start and end times, in observations (#/datapoints, not minutes), by default []

    Returns
    -------
    list of floats
        List of the average time between captures, where each item in this list
        corresponds to an entry in the input list intervals.
    """
    assert len(intervals) > 0
    capture_freqs = []
    for interval in intervals:
        start_obs, end_obs = interval
        interval_capture_counts = []
        windows = []
        for channel in captures_by_channel.keys():
            captures = captures_by_channel[channel]
            captures = raw_signal_utils.get_overlapping_regions(interval, captures)
            windows = capture_windows[channel]
            windows = raw_signal_utils.get_overlapping_regions(interval, windows)
            assert captures is not None
            assert windows is not None
            assert len(windows) > 0

            # TODO compute overlap with interval

            counts = calc_capture_freq(windows, captures)
            interval_capture_counts.append(counts)
        avg_capture_count = np.mean(interval_capture_counts)
        elapsed_time_obs = 0
        for window in windows:
            # For now, all channel windows identical
            elapsed_time_obs += window[1] - window[0]
        avg_capture_freq = avg_capture_count / (elapsed_time_obs / sampling_rate / 60.0)
        capture_freqs.append(avg_capture_freq)
    return capture_freqs


# assert len(intervals) > 0
#     capture_times = []
#     elapsed_time_obs = 0
#     for interval in intervals:
#         start_obs, end_obs = interval
#         # Compute capture freq for all channels separately, pooling results
#         interval_capture_times = []
#         for channel in captures_by_channel.keys():
#             captures = captures_by_channel[channel]
#             captures = raw_signal_utils.get_overlapping_regions(interval, captures)
#             blockages = blockages_by_channel[channel]
#             blockages = raw_signal_utils.get_overlapping_regions(interval, blockages)
#             windows = capture_windows[channel]
#             windows = raw_signal_utils.get_overlapping_regions(interval, windows)

#             assert captures is not None
#             assert blockages is not None
#             assert windows is not None
#             ts = calc_time_until_capture(windows, captures, blockages=blockages)
#             if ts is None:
#                 elapsed_time_obs += end_obs - start_obs
#             else:
#                 ts = [ts[0] + elapsed_time_obs] + ts[1:]
#                 interval_capture_times.extend(ts)
#                 elapsed_time_obs = 0
#         capture_times.append(np.mean(interval_capture_times))
#     return capture_times


def NTER_time_fit(time):
    """Convert time_between_captures to concentration per the NTER paper.

    Parameters
    ----------
    time : float
        Returned value from def calc_time_until_capture.


    Returns
    -------
    float
        Concentration
    """

    if time == -1:
        return 0
    conc = np.power(time / 20384.0, 1 / -0.96)
    if conc < 0:
        return 0
    return conc


def NTER_freq_fit(freq):
    """Convert capture_freq to concentration per the NTER paper.

    Parameters
    ----------
    freq : float
        Returned value from def calc_capture_freq.


    Returns
    -------
    float
        Concentration
    """

    if freq == -1:
        return 0
    conc = np.power(freq / 1.0263, 1 / 0.5239)
    if conc < 0:
        return 0
    return conc
