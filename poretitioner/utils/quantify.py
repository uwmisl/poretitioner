"""
===========
quantify.py
===========


This module contains functionality for quantifying nanopore captures.

"""
import re

import h5py
import numpy as np
import pandas as pd

from poretitioner import logger
from poretitioner.utils import classify, raw_signal_utils


def quantify_files(
    config,
    fast5_fnames,
    overwrite=False,
    filter_name=None,
    quant_method="time between captures",
    classified_only=False,
    interval_mins=None,
):
    # All fast5_fnames are assumed to be from the same run

    #
    # If a filter is specified in the config, use those reads only

    # The challenge here is that reads will be in different files. Need to read
    # in all the reads from all the files, sort them by channel and time, and
    # then quantify.

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

    # if quant_method == "time between captures":
    #     quantifier = calc_time_until_capture
    # elif quant_method == "capture freq":
    #     quantifier = calc_capture_freq
    # else:
    #     raise ValueError(f"This quantification method is not supported: {quant_method}")

    # TODO: Streaming opportunity -- shouldn't have to read all data at once.
    capture_arrays = []
    blockage_arrays = []

    for fast5_fname in fast5_fnames:
        with h5py.File(fast5_fname, "r") as f5:
            sampling_rate = raw_signal_utils.get_sampling_rate(f5)
            capture_array = get_capture_details_in_f5(f5, read_path, classification_path)
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
    interval_mins = 1
    if interval_mins is None:
        intervals = [(start_obs, end_obs)]
    else:
        interval_obs = interval_mins * 60 * sampling_rate
        intervals = list(range(interval_obs + start_obs, end_obs + 1, interval_obs))
        intervals = list(zip([start_obs] + intervals, intervals + [end_obs]))

    if quant_method == "time between captures":
        # TODO move this to its own function, taking in intervals as well
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

    elif quant_method == "capture freq":
        # TODO implement intervals
        capture_freqs = []
        for interval in intervals:
            start_obs, end_obs = interval
            interval_capture_counts = []
            for channel in captures_by_channel.keys():
                captures = captures_by_channel[channel]
                windows = capture_windows[channel]

                assert captures is not None
                assert windows is not None
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


def get_capture_windows_by_channel(fast5_fname):
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

    Note: called by "get_capture_time" and "get_capture_time_tseg"

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


def calc_capture_freq(capture_windows, captures, blockages=None):
    # local_logger = logger.getLogger()

    if captures is None or len(captures) == 0:
        return None

    n_captures = 0

    for capture_window_i, capture_window in enumerate(capture_windows):
        window_start, window_end = capture_window

        # Get all the captures within that window
        captures_in_window = raw_signal_utils.get_overlapping_regions(capture_window, captures)
        n_captures += len(captures_in_window)
    return n_captures


# # Getting Time Between Captures
# Returns list of average times until capture for given time intervals of a
# single run


def get_related_files(item, raw_file_dir="", capture_file_dir=""):
    # Temporary to prevent errors
    pass


def get_time_between_captures(
    filtered_file, time_interval=None, raw_file_dir="", capture_file_dir="", config_file=""
):
    """Get the average time between captures across all channels. Can be
    computed for the specified time interval, or across the entire run if not
    specified.

    Parameters
    ----------
    filtered_file : [type]
        [description]
    time_interval : [type], optional
        [description], by default None
    raw_file_dir : str, optional
        [description], by default ""
    capture_file_dir : str, optional
        [description], by default ""
    config_file : str, optional
        [description], by default ""

    Returns
    -------
    [type]
        [description]
    """
    # TODO: Implement logger best practices : https://github.com/uwmisl/poretitioner/issues/12
    local_logger = logger.getLogger("get_time_between_captures")

    # TODO : Implement capture fast5 I/O : https://github.com/uwmisl/poretitioner/issues/40

    # Retrieve raw file, unfiltered capture file, and config file names
    raw_file, capture_file = get_related_files(
        filtered_file, raw_file_dir=raw_file_dir, capture_file_dir=capture_file_dir
    )

    # Process raw file
    f5 = h5py.File(raw_file)
    # Find regions where voltage is normal
    voltage = f5.get("/Device/MetaData").value["bias_voltage"] * 5.0
    voltage_changes = raw_signal_utils.find_segments_below_threshold(voltage, -180)
    f5.close()

    # Process unfiltered captures file
    if capture_file.endswith(".pkl"):
        blockages = pd.read_pickle(capture_file)
    else:
        blockages = pd.read_csv(capture_file, index_col=0, header=0, sep="\t")

    # Process config file
    y = ""  # yaml_assistant.YAMLAssistant(config_file)
    run_name = re.findall(r"(run\d\d_.*)\..*", filtered_file)[0]
    good_channels = y.get_variable("fast5:good_channels:" + run_name)
    for i in range(0, len(good_channels)):
        good_channels[i] = "Channel_" + str(good_channels[i])
    local_logger.info("Number of Channels: " + str(len(good_channels)))

    # Process filtered captures file
    captures = pd.read_csv(filtered_file, index_col=0, header=0, sep="\t")

    # Break run into time segments based on time interval (given in minutes).
    # If no time interval given then just take average time between captures of
    # entire run
    if time_interval:
        # TODO un-hardcode 10k number -- look up from fast5 file directly.
        time_segments = range(
            time_interval * 60 * 10000, len(voltage) + 1, time_interval * 60 * 10000
        )
    else:
        time_segments = [len(voltage)]

    # Calculate Average Time Between Captures for Each Time Segment #

    # Tracks time elapsed (no capture) for each channel across time segments
    time_elapsed = [0 for x in range(0, len(good_channels))]
    # List of mean capture times across all channels for each timepoint
    timepoint_captures = []
    captures_count = []  # Number of captures across all channels for each
    # timepoint
    checkpoint = 0
    for timepoint in time_segments:
        voltage_changes_segment = []
        blockage_ends, blockage_starts = "", ""
        # Find open voltage regions that start within this time segment
        for voltage_region in voltage_changes:
            if voltage_region[0] < timepoint and voltage_region[0] >= checkpoint:
                voltage_changes_segment.append(voltage_region)
        # If this time segment contains open voltage regions...
        if voltage_changes_segment:
            # End of last voltage region in tseg
            end_voltage_seg = voltage_changes_segment[len(voltage_changes_segment) - 1][1]
            capture_times = []  # Master list of all capture times from this seg
            # Loop through all good channels and get captures times from each
            for i, channel in enumerate(good_channels):
                channel_blockages = blockages[blockages.channel == channel]
                blockage_exists = False
                # If there are any blockages in this tseg (includes both
                # non-captures and captures)
                blockage_segment = channel_blockages[
                    np.logical_and(
                        channel_blockages.start_obs <= end_voltage_seg,
                        channel_blockages.start_obs > checkpoint,
                    )
                ]
                if not channel_blockages.empty and not blockage_segment.empty:
                    blockage_exists = True
                    blockage_starts = list(blockage_segment.start_obs)
                    blockage_ends = list(blockage_segment.end_obs)
                    # TODO temporary fix; may be equivalent to blockage_segment
                    blockages = list(zip(blockage_starts, blockage_ends))

                channel_captures = captures[captures.channel == channel]
                # Check that channel actually has captures in this tseg
                captures_segment = channel_captures[
                    np.logical_and(
                        channel_captures.start_obs <= end_voltage_seg,
                        channel_captures.start_obs > checkpoint,
                    )
                ]
                if not channel_captures.empty and not captures_segment.empty:

                    time_until_capture = calc_time_until_capture(
                        voltage_changes_segment, captures_segment, blockages=blockages
                    )
                    # Add time since channel's last capture from previous
                    # tsegs to time until first capture in current tseg
                    time_until_capture[0] += time_elapsed[i]

                    # Update to new time elapsed (time from end of last capture
                    # in this tseg to end of tseg minus blockages)
                    time_elapsed[i] = 0
                    voltage_ix = 0
                    while voltage_ix < len(voltage_changes_segment):
                        if voltage_changes_segment[voltage_ix][0] > captures_segment[-1].end_obs:
                            time_elapsed[i] += np.sum(
                                calc_time_until_capture(
                                    voltage_changes_segment[voltage_ix:], blockages
                                )
                            )
                            break
                        voltage_ix += 1
                    time_elapsed[i] += end_voltage_seg - blockage_ends[-1]

                    capture_times.extend(time_until_capture)
                else:
                    # No captures but still blockages, so add duration of open
                    # voltage regions minus blockages to time elapsed
                    if blockage_exists:
                        time_elapsed[i] += np.sum(
                            calc_time_until_capture(
                                voltage_changes_segment, blockage_starts, blockage_ends
                            )
                        )
                        time_elapsed[i] += end_voltage_seg - blockage_ends[-1]
                    # No captures or blockages for channel in this tseg, so add
                    # total duration of open voltage regions to time elapsed
                    else:
                        time_elapsed[i] += np.sum(
                            [
                                voltage_region[1] - voltage_region[0]
                                for voltage_region in voltage_changes_segment
                            ]
                        )
            if capture_times:
                timepoint_captures.append(np.mean(capture_times))
            else:
                timepoint_captures.append(-1)

            captures_count.append(len(capture_times))
            checkpoint = end_voltage_seg
        else:
            local_logger.warning(
                "No open voltage region in time segment ["
                + str(checkpoint)
                + ", "
                + str(timepoint)
                + "]"
            )
            timepoint_captures.append(-1)
            checkpoint = timepoint

    local_logger.info("Number of Captures: " + str(captures_count))
    return timepoint_captures


# # Getting Capture Frequency
# Returns list of capture frequencies (captures/channel/min) for each time
# interval.
# Time intervals must be equal duration and start from zero!


def get_capture_freq(
    filtered_file, time_interval=None, raw_file_dir="", capture_file_dir="", config_file=""
):
    # TODO: Implement logger best practices : https://github.com/uwmisl/poretitioner/issues/12
    local_logger = logger.getLogger()
    # TODO : Implement capture fast5 I/O : https://github.com/uwmisl/poretitioner/issues/40

    # Retrieve raw file and config file names
    raw_file, capture_file = get_related_files(
        filtered_file, raw_file_dir=raw_file_dir, capture_file_dir=capture_file_dir
    )

    # Process raw file
    f5 = h5py.File(raw_file)
    # Find regions where voltage is normal
    voltage = f5.get("/Device/MetaData").value["bias_voltage"] * 5.0
    voltage_changes = []  # find_segments_below_threshold(voltage, -180)
    f5.close()

    # Process config file
    # y = YAMLAssistant(config_file)
    y = ""
    # [-11:-4] gives run_seg (i.e. "run01_a")
    good_channels = y.get_variable("fast5:good_channels:" + filtered_file[-11:-4])
    for i in range(0, len(good_channels)):
        good_channels[i] = "Channel_" + str(good_channels[i])
    local_logger.info("Number of Channels: " + str(len(good_channels)))

    # Process filtered captures file
    captures = pd.read_csv(filtered_file, index_col=0, header=0, sep="\t")

    # Break run into time segments based on time interval (given in minutes).
    # If no time interval given then just take average time between captures of
    # entire run
    if time_interval:
        time_segments = range(
            time_interval * 60 * 10000, len(voltage) + 1, time_interval * 60 * 10000
        )
    else:
        time_segments = [len(voltage)]

    # Calculate Capture Frequency for Each Time Segment #

    all_capture_freq = []  # List of capture frequencies for each timepoint
    checkpoint = 0
    for timepoint in time_segments:
        voltage_changes_segment = []
        # Find open voltage regions that start within this time segment
        for voltage_region in voltage_changes:
            if voltage_region[0] < timepoint and voltage_region[0] >= checkpoint:
                voltage_changes_segment.append(voltage_region)

        # If this time segment contains open voltage regions...
        if voltage_changes_segment:
            # End of last voltage region in tseg
            end_voltage_seg = voltage_changes_segment[len(voltage_changes_segment) - 1][1]
            # List of capture counts for each channel from this tseg (length of
            # list = # of channels)
            capture_counts = []
            # Loop through all good channels and get captures times from each
            for i, channel in enumerate(good_channels):
                channel_captures = captures[captures.channel == channel]
                # Check that channel actually has captures and add the # of
                # captures in this tseg to capture_counts
                if not channel_captures.empty:
                    capture_counts.append(
                        len(
                            channel_captures[
                                np.logical_and(
                                    channel_captures.start_obs <= end_voltage_seg,
                                    channel_captures.start_obs > checkpoint,
                                )
                            ]
                        )
                    )
                else:
                    capture_counts.append(0)
            all_capture_freq.append(np.mean(capture_counts) / (time_segments[0] / 600_000.0))
            checkpoint = end_voltage_seg
        else:
            local_logger.warning(
                "No open voltage region in time segment ["
                + str(checkpoint)
                + ", "
                + str(timepoint)
                + "]"
            )
            all_capture_freq.append(0)
            checkpoint = timepoint

    return all_capture_freq


# Calibration Curves


def NTER_time_fit(time):
    # TODO : Document the hardcoded values : https://github.com/uwmisl/poretitioner/issues/44

    if time == -1:
        return 0
    conc = np.power(time / 20384.0, 1 / -0.96)
    if conc < 0:
        return 0
    return conc


def NTER_freq_fit(freq):
    # TODO : Document the hardcoded values : https://github.com/uwmisl/poretitioner/issues/44

    if freq == -1:
        return 0
    conc = np.power(freq / 1.0263, 1 / 0.5239)
    if conc < 0:
        return 0
    return conc
