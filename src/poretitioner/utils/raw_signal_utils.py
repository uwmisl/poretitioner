"""
===================
raw_signal_utils.py
===================

This module contains general purpose utility functions for accessing raw
nanopore current within fast5 files, and for manipulating the raw current.

"""

import re
from collections import namedtuple
from typing import List

import h5py
import numpy as np

__all__ = ["judge_channels"]


def natkey(string_):
    """Natural sorting key -- sort strings containing numerics so numerical
    ordering is honored/preserved.

    Parameters
    ----------
    string_ : str
        String to be converted for sorting.

    Returns
    -------
    Mixed-type list of str and int
        List to be used as a key for sorting.
    """
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", string_)]


def judge_channels(bulk_f5_fname, expected_open_channel=235):
    """Judge channels based on quality of current. If the current is too low,
    the channel is probably off (bad), etc.

    Parameters
    ----------
    bulk_f5_fname : str
        Bulk fast5 filename (cannot be capture/read fast5).
    expected_open_channel : int, optional
        Approximate estimate of the open channel current value., by default 235

    Returns
    -------
    list of ints
        List of good channels. NOTE: Channels are 1-indexed, The first channel will be 1, not 0.
    """
    f5 = h5py.File(name=bulk_f5_fname)
    channels = list(f5.get("Raw").keys())
    channels.sort(key=natkey)
    good_channels = []
    for channel in channels:
        channel_no = int(re.findall(r"Channel_(\d+)", channel)[0])
        raw = get_scaled_raw_for_channel(f5, channel_no)

        # Case 1: Channel might not be totally off, but has little variance
        if np.abs(np.mean(raw)) < 20 and np.std(raw) < 50:
            continue

        # Case 2: Neither the median or 75th percentile value contains the
        #         open pore current.
        # if expected_open_channel is not None:
        #     sorted_raw = sorted(raw)
        #     len_raw = len(raw)
        #     q_50 = int(len_raw / 2)
        #     q_75 = int(len_raw * 3 / 4)
        #     median_outside_range = np.abs(sorted_raw[q_50] - expected_open_channel) > 25
        #     upper_outside_range = np.abs(sorted_raw[q_75] - expected_open_channel) > 25
        #     if median_outside_range and upper_outside_range:
        #         continue

        # Case 3: The channel is off
        off_regions = find_signal_off_regions(raw, current_range=100)
        off_points = []
        for start, end in off_regions:
            off_points.extend(range(start, end))
        if len(off_points) + 50000 > len(raw):
            continue

        # Case 4: The channel is assumed to be good
        good_channels.append(channel_no)
    return good_channels


def get_overlapping_regions(window, regions):
    """get_overlapping_regions

    Finds all of the regions in the given list that overlap with the window.
    Needs to have at least one overlapping point; cannot be just adjacent.
    Incomplete overlaps are returned.

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
