"""
===========
quantify.py
===========


This module contains functionality for quantifying nanopore captures.

"""
import logging
import os
import re

import h5py
import numpy as np
import pandas as pd

from .raw_signal_utils import find_segments_below_threshold
from .yaml_assistant import YAMLAssistant

# # Retrieving Related Data Files from Filtered File or Capture File
# Returns list containing path names of `[raw_file, capture_file, config_file]`
# corresponding to the passed in file name. Passed in file can be either filter
# file or capture file (total captures).


def get_related_files(input_file, raw_file_dir="", capture_file_dir=""):
    """ TODO : Deprecate! : https://github.com/uwmisl/poretitioner/issues/40

    Find files matching the input file in the data directory tree.

    Parameters
    ----------
    input_file : string
        # TODO ???
        Seems like this can be the filtered or unfiltered captures file.
    raw_file_dir : str, optional
        Directory to search for the capture files (for us that's
        /disk1/pore_data/MinION_raw_data_YYYYMMDD/), by default, the cwd.
    capture_file_dir : str, optional
        Directory to search for the capture files (for us that's
        /disk1/pore_data/segmented/peptides/YYYYMMDD/), by default, the cwd.

    Returns
    -------
    Iterable of filenames (strings)
        The raw file (fast5) and capture file (unfiltered).
    """
    logger = logging.getLogger("get_related_files")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    logger.debug(input_file)
    logger.debug(raw_file_dir)
    logger.debug(capture_file_dir)

    run_name = re.findall(r"(run\d\d_.*)\..*", input_file)[0]  # e.g. "run01_a"
    logger.debug(run_name)

    assert len(raw_file_dir) > 0
    raw_file = [x for x in os.listdir(raw_file_dir) if run_name in x][0]

    assert len(capture_file_dir) > 0
    if input_file.endswith(".csv"):
        # Given file is the filtered file and we're looking for the capture file
        filtered_file = input_file
        capture_file = [x for x in os.listdir(capture_file_dir) if x.endswith(run_name + ".pkl")][
            0
        ]
    elif input_file.endswith(".pkl"):
        # Given file is the capture file and filtered file is unspecified
        capture_file = input_file
        filtered_file = "Unspecified"
    else:
        logger.error("Invalid file name")
        return

    logger.info("Filter File: " + filtered_file)
    raw_file = os.path.join(raw_file_dir, raw_file)
    logger.info("Raw File: " + raw_file)
    capture_file = os.path.join(capture_file_dir, capture_file)
    logger.info("Capture File: " + capture_file)

    return raw_file, capture_file


def get_overlapping_regions(window, regions):
    """get_overlapping_regions

    Finds all of the regions in the given list that overlap with the window.
    Needs to have at least one overlapping point; cannot be just adjacent.
    Incomplete overlaps are returned.

    # TODO move to raw_signal_utils -- general purpose signal fn not specific to quant

    Parameters
    ----------
    window : tuple of numerics (start, end)
        Specifies the start and end points of the desired overlap.
    regions : list of tuples of numerics [(start, end), ...]
        Start and end points to check for overlap with the specified window.
        All regions are assumed to be mutually exclusive and sorted in
        ascending order.

    Returns
    -------
    overlapping_regions : list of tuples of numerics [(start, end), ...]
        Regions that overlap with the window in whole or part, returned in
        ascending order.
    """
    window_start, window_end = window
    overlapping_regions = []
    for region in regions:
        region_start, region_end = region
        if region_start >= window_end:
            # Region comes after the window, we're done searching
            break
        elif region_end <= window_start:
            # Region ends before the window starts
            continue
        elif region_end > window_start or region_start >= window_start:
            # Overlapping
            overlapping_regions.append(region)
        else:
            e = f"Shouldn't have gotten here! Region: {region}, Window: {window}."
            raise Exception(e)
    return overlapping_regions


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
    logger = logging.getLogger("calc_time_until_capture")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    all_capture_times = []

    if captures is None or len(captures) == 0:
        return None

    elapsed_time_until_capture = 0
    for capture_window_i, capture_window in enumerate(capture_windows):
        # Get all the captures & blockages within that window
        captures_in_window = get_overlapping_regions(capture_window, captures)
        if blockages is not None:
            blockages_in_window = get_overlapping_regions(capture_window, blockages)
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


# # Getting Time Between Captures
# Returns list of average times until capture for given time intervals of a
# single run


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
    logger = logging.getLogger("get_time_between_captures")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # TODO : Implement capture fast5 I/O : https://github.com/uwmisl/poretitioner/issues/40

    # Retrieve raw file, unfiltered capture file, and config file names
    raw_file, capture_file = get_related_files(
        filtered_file, raw_file_dir=raw_file_dir, capture_file_dir=capture_file_dir
    )

    # Process raw file
    f5 = h5py.File(raw_file)
    # Find regions where voltage is normal
    voltage = f5.get("/Device/MetaData").value["bias_voltage"] * 5.0
    voltage_changes = find_segments_below_threshold(voltage, -180)
    f5.close()

    # Process unfiltered captures file
    if capture_file.endswith(".pkl"):
        blockages = pd.read_pickle(capture_file)
    else:
        blockages = pd.read_csv(capture_file, index_col=0, header=0, sep="\t")

    # Process config file
    y = YAMLAssistant(config_file)
    run_name = re.findall(r"(run\d\d_.*)\..*", filtered_file)[0]
    good_channels = y.get_variable("fast5:good_channels:" + run_name)
    for i in range(0, len(good_channels)):
        good_channels[i] = "Channel_" + str(good_channels[i])
    logger.info("Number of Channels: " + str(len(good_channels)))

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
            logger.warn(
                "No open voltage region in time segment ["
                + str(checkpoint)
                + ", "
                + str(timepoint)
                + "]"
            )
            timepoint_captures.append(-1)
            checkpoint = timepoint

    logger.info("Number of Captures: " + str(captures_count))
    return timepoint_captures


# # Getting Capture Frequency
# Returns list of capture frequencies (captures/channel/min) for each time
# interval.
# Time intervals must be equal duration and start from zero!


def get_capture_freq(
    filtered_file, time_interval=None, raw_file_dir="", capture_file_dir="", config_file=""
):
    # TODO: Implement logger best practices : https://github.com/uwmisl/poretitioner/issues/12
    logger = logging.getLogger("get_capture_freq")
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # TODO : Implement capture fast5 I/O : https://github.com/uwmisl/poretitioner/issues/40

    # Retrieve raw file and config file names
    raw_file, capture_file = get_related_files(
        filtered_file, raw_file_dir=raw_file_dir, capture_file_dir=capture_file_dir
    )

    # Process raw file
    f5 = h5py.File(raw_file)
    # Find regions where voltage is normal
    voltage = f5.get("/Device/MetaData").value["bias_voltage"] * 5.0
    voltage_changes = find_segments_below_threshold(voltage, -180)
    f5.close()

    # Process config file
    y = YAMLAssistant(config_file)
    # [-11:-4] gives run_seg (i.e. "run01_a")
    good_channels = y.get_variable("fast5:good_channels:" + filtered_file[-11:-4])
    for i in range(0, len(good_channels)):
        good_channels[i] = "Channel_" + str(good_channels[i])
    logger.info("Number of Channels: " + str(len(good_channels)))

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
            logger.warn(
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
