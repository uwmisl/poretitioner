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

from .core import Channel, ChannelCalibration, Window

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
        List of good channels.
    """
    f5 = h5py.File(name=bulk_f5_fname)
    channels = f5.get("Raw").keys()
    channels.sort(key=natkey)
    good_channels = []
    for channel in channels:
        i = int(re.findall(r"Channel_(\d+)", channel)[0])
        raw = get_scaled_raw_for_channel(f5, channel)

        # Case 1: Channel might not be totally off, but has little variance
        if np.abs(np.mean(raw)) < 20 and np.std(raw) < 50:
            continue

        # Case 2: Neither the median or 75th percentile value contains the
        #         open pore current.
        if expected_open_channel is not None:
            sorted_raw = sorted(raw)
            len_raw = len(raw)
            q_50 = len_raw / 2
            q_75 = len_raw * 3 / 4
            median_outside_rng = np.abs(sorted_raw[q_50] - expected_open_channel) > 25
            upper_outside_rng = np.abs(sorted_raw[q_75] - expected_open_channel) > 25
            if median_outside_rng and upper_outside_rng:
                continue

        # Case 3: The channel is off
        off_regions = find_signal_off_regions(raw, current_range=100)
        off_points = []
        for start, end in off_regions:
            off_points.extend(range(start, end))
        if len(off_points) + 50000 > len(raw):
            continue

        # Case 4: The channel is assumed to be good
        good_channels.append(i)
    return good_channels
