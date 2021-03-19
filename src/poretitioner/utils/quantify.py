"""
===========
quantify.py
===========


This module contains functionality for quantifying nanopore captures.

"""
import re
from pathlib import PurePosixPath
from typing import List, Optional

import h5py
import numpy as np
from src.poretitioner.fast5s import CaptureFile
from src.poretitioner.logger import Logger, getLogger
from src.poretitioner.utils.classify import ClassifierFile
from src.poretitioner.utils.core import PathLikeOrString, Window, WindowsByChannel


class QuantifyCaptureFile(ClassifierFile):
    def __init__(
        self, capture_filepath: PathLikeOrString, logger: Logger = getLogger()
    ):
        super().__init__(capture_filepath, logger=logger)
        # self.classifier_path = PurePosixPath(self.ROOT, "Classification", classifier_details.model)

    def get_capture_details(self, read_path=None, classification_model=None):
        """Get capture times & channel info for all reads in the fast5 at read_path.
        If a classification_path is specified, get the label for each read too.

        Parameters
        ----------
        read_path : str, optional
            Location of the reads to retrieve, by default None (if None, defaults
            to "/" for the read location.)
        classification_model : str, optional
            Name of the classifier to use the labels of, by default None (if None, no
            labels are returned)

        Returns
        -------
        np.array
            Array containing read_id, capture/read start & end, channel number, and
            the classification label (if applicable). Shape is (5 x n_reads).
        """
        f5 = self.f5

        read_path = read_path if read_path is not None else self.ROOT
        read_h5group_names = f5.get(read_path)

        captures_in_f5 = []
        for read_id in self.reads:
            capture_metadata = self.get_capture_metadata_for_read(read_id)
            capture_start = capture_metadata.start_time_local
            capture_end = capture_start + capture_metadata.duration
            channel_number = capture_metadata.channel_number
            self.get_classification_for_read(classification_model, read_id)
            # classify.get_classification_for_read(f5, read_id, classification_path)

        if read_h5group_names is None:
            return None

        for grp_name in read_h5group_names:
            if "read" not in grp_name:
                continue
            read_id = re.findall(r"read_(.*)", grp_name)[0]
            grp = f5.get(grp_name)
            a = grp["Signal"].attrs
            capture_start = capture_metadata.start_time_local
            capture_end = capture_start + capture_metadata.duration

            a = grp["channel_id"].attrs
            channel_number = capture_metadata.channel_number

            if classification_path:
                (
                    pred_class,
                    prob,
                    assigned_class,
                    passed_classification,
                ) = self.get_classification_for_read(read_id, classification_path)
            else:
                assigned_class = None

            captures_in_f5.append(
                (read_id, capture_start, capture_end, channel_number, assigned_class)
            )

        return np.array(captures_in_f5)


def quantify_files(
    config,
    capture_filepaths,
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
    capture_filepaths : list of str
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

    if len(capture_filepaths) == 0:
        raise ValueError("No fast5 files specified (len(capture_filepaths) == 0).")

    # The challenge here is that reads will be in different files. Need to read
    # in all the reads from all the files, sort them by channel and time, and
    # then quantify.
    for capture_file in capture_filepaths:
        with CaptureFile(capture_file) as capture:
            sampling_rate = capture.sampling_rate

            capture_array = get_capture_details_in_f5(
                capture, read_path=read_path, classification_path=classification_path
            )
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
                blockage_array = get_capture_details_in_f5(
                    capture,
                    read_path=read_path,
                    classification_path=classification_path,
                )
            if len(blockage_array) > 0:
                blockage_arrays.append(blockage_array)

    captures_by_channel = sort_captures_by_channel(np.vstack(capture_arrays))
    blockages_by_channel = sort_captures_by_channel(np.vstack(blockage_arrays))

    del capture_arrays, blockage_arrays

    # The capture windows are stored identically in each read fast5, so no need for loop
    capture_windows = get_capture_windows_by_channel(capture_file)
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


def get_capture_windows_by_channel(capture_file):
    """Extract capture windows (regions where the voltage is appropriate to accept
    a captured peptide or other analyte) from a fast5 file, in a dictionary and
    sorted by channel

    Parameters
    ----------
    capture_file : string
        Location of a fast5 file.

    Returns
    -------
    dict
        {channel_no: [(window_start, window_end), ...]}
    """
    base_path = "/Meta/Segmentation/capture_windows"
    windows_by_channel = {}
    with h5py.File(capture_file, "r") as f5:
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


def get_capture_details_in_f5(
    capture: CaptureFile, read_path=None, classification_path=None
):
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
    f5 = capture.f5
    read_path = read_path if read_path is not None else "/"
    read_h5group_names = f5.get(read_path)

    captures_in_f5 = []
    for read in capture.reads:
        capture_metadata = capture.get_capture_metadata_for_read(read)

    if read_h5group_names is None:
        return None

    for grp_name in read_h5group_names:
        if "read" not in grp_name:
            continue
        read_id = re.findall(r"read_(.*)", grp_name)[0]
        grp = f5.get(grp_name)
        a = grp["Signal"].attrs
        capture_start = capture_metadata.start_time_local
        capture_end = capture_start + capture_metadata.duration

        a = grp["channel_id"].attrs
        channel_number = capture_metadata.channel_number

        if classification_path:
            (
                pred_class,
                prob,
                assigned_class,
                passed_classification,
            ) = classify.get_classification_for_read(f5, read_id, classification_path)
        else:
            assigned_class = None

        captures_in_f5.append(
            (read_id, capture_start, capture_end, channel_number, assigned_class)
        )

    return np.array(captures_in_f5)


# TODO: Katie Q: I could be wrong, this feels more general than 'quantify', might this method be appropriate for "core.py"?
def calc_time_until_capture(
    open_pore_windows: List[Window],
    captures: List[Window],
    blockages: Optional[List[Window]] = None,
) -> List[float]:
    """
    Finds all times between captures from a single channel. This is defined
    as the open pore time from the end of the previous capture to the
    current capture. Includes subtracting other non-capture blockages since
    those blockages reduce the amount of overall open pore time.

    Parameters
    ----------
    open_pore_windows : List[Window]
        Regions of current where the nanopore is available to accept a
        capture. (I.e., is in a "normal" voltage state.) [(start, end), ...]
    captures : List[Window]
        Regions of current where a capture is residing in the pore. The
        function is calculating time between these values (minus blockages).
    blockages : Optional[List[Window]], optional
        Regions of current where the pore is blocked by any capture or non-
        capture. These are removed from the time between captures, if
        specified. None by default.

    Returns
    -------
    List[float]
        List of all capture times from a single channel.
    """
    # TODO: Implement logger best practices : https://github.com/uwmisl/poretitioner/issues/12
    # local_logger = logger.getLogger()
    if captures is None or len(captures) == 0:
        return []

    all_capture_times = []

    # TODO Katie Q: Would you mind double-checking my changes here?
    # (e.g. that I'm getting the window start and ends right, and haven't introduced bugs into your original code)
    elapsed_time_until_capture = 0
    for capture_window_i, capture_window in enumerate(open_pore_windows):
        # Get all the captures & blockages within that window
        captures_in_window = capture_window.overlaps(captures)
        if blockages is not None:
            blockages_in_window = capture_window.overlaps(blockages)
        else:
            blockages_in_window = []
        # If there are no captures in the window, add the window to the elapsed
        # time and subtract any blockages.
        if len(captures_in_window) == 0:
            elapsed_time_until_capture += capture_window.duration
            for blockage in blockages_in_window:
                elapsed_time_until_capture -= blockage.duration
            continue
        # If there's a capture in the window, add the partial window to the
        # elapsed time. Subtract blockages that came before the capture.
        else:
            last_capture_end = capture_window.start
            for capture_i, capture in enumerate(captures_in_window):
                elapsed_time_until_capture += capture.start - last_capture_end
                for blockage in blockages_in_window:
                    # Blockage must start after the last capture ended and
                    # finish before the next capture starts; otherwise skip
                    if (
                        blockage.start >= last_capture_end
                        and blockage.end < capture.start
                    ):
                        elapsed_time_until_capture -= blockage.duration
                        blockages.pop(0)
                all_capture_times.append(elapsed_time_until_capture)
                # Reset elapsed time.
                elapsed_time_until_capture = max(capture_window.end - capture.end, 0)
                captures.pop(0)
                last_capture_end = capture.end
    return all_capture_times


# TODO: Katie Q: I could be wrong, this feels more general than 'quantify', might this method be appropriate for "core.py"?
def calc_time_until_capture_intervals(
    capture_windows: List[Window],
    captures_by_channel: WindowsByChannel,
    blockages_by_channel: WindowsByChannel = None,
    intervals: Optional[List[Window]] = None,
) -> List[float]:
    """Compute time_until_capture across specified intervals of time.

    Parameters
    ----------
    capture_windows : List[Window]
        Regions of current where the nanopore is available to accept a
        capture. (I.e., is in a "normal" voltage state.) [(start, end), ...]
    captures_by_channel : WindowsByChannel of captures e.{channel_no: [(start, end), ...]}
        Regions of current where a capture is residing in the pore. The
        function is calculating time between these values (minus blockages).
    blockages_by_channel : dictionary of blockages {channel_no: [(start, end), ...]}, optional
        Regions of current where the pore is blocked by any capture or non-
        capture. These are removed from the time between captures, if
        specified.
    intervals : List[Window], optional
        List of start and end times, in observations (#/datapoints, not minutes), by default []

    Returns
    -------
    list of floats
        List of the average time between captures, where each item in this list
        corresponds to an entry in the input list intervals.
    """
    intervals: List[Window] = (
        intervals if intervals is not None else []
    )  # Defaults to empty list.

    assert len(intervals) > 0
    capture_times = []
    elapsed_time_obs = 0
    for interval in intervals:
        start_obs, end_obs = interval
        # Compute capture freq for all channels separately, pooling results
        interval_capture_times = []
        for channel in captures_by_channel.keys():
            captures = captures_by_channel[channel]
            interval.overlaps(captures)
            blockages = blockages_by_channel[channel]
            interval.overlaps(blockages)
            windows = capture_windows[channel]
            interval.overlaps(windows)

            assert captures is not None
            assert blockages is not None
            assert windows is not None
            ts = calc_time_until_capture(windows, captures, blockages=blockages)
            if len(ts) < 1:
                elapsed_time_obs += end_obs - start_obs
            else:
                ts = [ts[0] + elapsed_time_obs] + ts[1:]
                interval_capture_times.extend(ts)
                elapsed_time_obs = 0
        capture_times.append(np.mean(interval_capture_times))
    return capture_times


# TODO: Katie Q: I could be wrong, this feels more general than 'quantify', might this method be appropriate for "core.py"?
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
        captures_in_window = capture_window.overlaps(captures)
        n_captures += len(captures_in_window)
    return n_captures


# TODO: Katie Q: I could be wrong, this feels more general than 'quantify', might this method be appropriate for "core.py"?


def calc_capture_freq_intervals(
    capture_windows, captures_by_channel, intervals=[], sampling_rate=10000
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
            captures = interval.overlaps(captures)
            windows = capture_windows[channel]
            windows = interval.overlaps(windows)
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
#             if len(ts) < 1:
#                 elapsed_time_obs += end_obs - start_obs
#             else:
#                 ts = [ts[0] + elapsed_time_obs] + ts[1:]
#                 interval_capture_times.extend(ts)
#                 elapsed_time_obs = 0
#         capture_times.append(np.mean(interval_capture_times))
#     return capture_times

# TODO: Katie Q: Is this still used anywhere? Will it be?
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


# TODO: Katie Q: Is this still used anywhere? Will it be?
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
