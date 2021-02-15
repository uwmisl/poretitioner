"""
===================
raw_signal_utils.py
===================

This module contains general purpose utility functions for accessing raw
nanopore current within fast5 files, and for manipulating the raw current.

"""

import re

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


def compute_fractional_blockage(scaled_raw, open_channel):
    """Converts a raw nanopore signal (in units of pA) to fractionalized current
    in the range (0, 1).

    A value of 0 means the pore is fully blocked, and 1 is fully open.

    Parameters
    ----------
    scaled_raw : array
        Array of nanopore current values in units of pA.
    open_channel : float
        Open channel current value (pA).

    Returns
    -------
    array of floats
        Array of fractionalized nanopore current in the range (0, 1)
    """
    scaled_raw = np.frombuffer(scaled_raw, dtype=float)
    scaled_raw /= open_channel
    scaled_raw = np.clip(scaled_raw, a_max=1.0, a_min=0.0)
    return scaled_raw


def get_fractional_blockage(
    f5, channel_no, start=None, end=None, open_channel_guess=220, open_channel_bound=15
):
    """Retrieve the scaled raw signal for the channel, compute the open pore
    current, and return the fractional blockage for that channel.

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    channel_no : int
        Channel number for which to retrieve raw signal.
    start : int, optional
        Retrieve a slice of current starting at this time point, by default None
        None will retrieve data starting from the beginning of available data.
    end : int, optional
        Retrieve a slice of current starting at this time point, by default None
        None will retrieve all data at the end of the array.
    open_channel_guess : int, optional
        Approximate estimate of the open channel current value, by default 220
    open_channel_bound : int, optional
        Approximate estimate of the variance in open channel current value from
        channel to channel (AKA the range to search), by default 15

    Returns
    -------
    Numpy array (float)
        Fractionalized current from the specified input channel.
    """
    if type(f5) is not h5py._hl.files.File:
        raise ValueError("f5 must be an open h5py.File object.")
    signal = get_scaled_raw_for_channel(f5, channel_no, start=start, end=end)
    open_channel = find_open_channel_current(signal, open_channel_guess, bound=open_channel_bound)
    if open_channel is None:
        # TODO add logging here to give the reason for returning None
        return None
    frac = compute_fractional_blockage(signal, open_channel)
    return frac


def get_local_fractional_blockage(
    f5, open_channel_guess=220, open_channel_bound=15, channel=None, local_window_sz=1000
):
    """Retrieve the scaled raw signal for the channel, compute the open pore
    current, and return the fractional blockage for that channel."""
    if type(f5) is not h5py._hl.files.File:
        raise ValueError("f5 must be an open h5py.File object.")
    signal = get_scaled_raw_for_channel(f5, channel=channel)
    open_channel = find_open_channel_current(signal, open_channel_guess, bound=open_channel_bound)
    if open_channel is None:
        # print("open pore is None")
        return None

    frac = np.zeros(len(signal))
    for start in range(0, len(signal), local_window_sz):
        end = start + local_window_sz
        local_chunk = signal[start:end]
        local_open_channel = find_open_channel_current(
            local_chunk, open_channel, bound=open_channel_bound
        )
        if local_open_channel is None:
            local_open_channel = open_channel
        frac[start:end] = compute_fractional_blockage(local_chunk, local_open_channel)
    return frac


def get_voltage(f5, start=None, end=None):
    """Retrieve the bias voltage from a bulk fast5 file.

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    start : int
        Start point  # TODO

    Returns
    -------
    float
        Voltage (mV).

    Raises
    ------
    ValueError
        Raised if f5 is not an h5py.File object.
    """
    if type(f5) is not h5py._hl.files.File:
        raise ValueError("f5 must be an open h5py.File object.")
    voltage = f5.get("/Device/MetaData")["bias_voltage"][start:end] * 5.0
    return voltage


def get_sampling_rate(f5):
    """Retrieve the sampling rate from a bulk fast5 file. Units: Hz.

    Also referred to as the sample rate, sample frequency, or sampling
    frequency.

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.

    Returns
    -------
    int
        Sampling rate

    Raises
    ------
    ValueError
        Raised if f5 is not an h5py.File object.
    """
    if type(f5) is not h5py._hl.files.File:
        raise ValueError("f5 must be an open h5py.File object.")
    try:
        sample_rate = int(f5.get("Meta").attrs["sample_rate"])
    except KeyError:
        sample_rate = int(f5.get("/Meta/context_tags").attrs["sample_frequency"])
    return sample_rate


def get_fractional_blockage_for_read(f5, read_id, start=None, end=None):
    """Retrieve the scaled raw signal for the specified read.

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    read_id : str
        read_id to retrieve fractionalized current.

    Returns
    -------
    Numpy array (float)
        Fractionalized current from the specified read_id.

    Raises
    ------
    ValueError
        Raised if f5 is not an h5py.File object.
    """
    if type(f5) is not h5py._hl.files.File:
        raise ValueError("f5 must be an open h5py.File object.")
    signal = get_scaled_raw_for_read(f5, read_id, start=start, end=end)
    if "read" in read_id:
        channel_path = f"{read_id}/channel_id"
    else:
        channel_path = f"read_{read_id}/channel_id"
    open_channel = f5.get(channel_path).attrs["open_channel_pA"]
    frac = compute_fractional_blockage(signal, open_channel)
    return frac


def get_raw_signal_for_read(f5, read_id, start=None, end=None):
    """Retrieve raw signal from open fast5 file for the specified read_id.

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    read_id : str
        Read id to retrieve raw signal. Can be formatted as a path ("read_xxx...")
        or just the read id ("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").

    Returns
    -------
    Numpy array
        Array representing sampled nanopore current.

    Raises
    ------
    ValueError
        Raised if f5 is not an h5py.File object.
        Raised if the path to the signal is not present in f5.
    """
    if type(f5) is not h5py._hl.files.File:
        raise ValueError("f5 must be an open h5py.File object.")
    if "read" in read_id:
        signal_path = f"/{read_id}/Signal"
    else:
        signal_path = f"/read_{read_id}/Signal"
    if signal_path not in f5:
        raise ValueError(f"Could not find path in fast5 file: {signal_path}, {f5.filename}")
    if signal_path in f5:
        raw = f5.get(signal_path)[start:end]
        return raw
    else:
        raise ValueError(f"Path {signal_path} not in fast5 file.")


def get_scaled_raw_for_read(f5, read_id, start=None, end=None):
    """Retrieve raw signal from open fast5 file, scaled to pA units.

    Note: using UK sp. of digitization for consistency w/ file format

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    read_id : str
        Read id to retrieve raw signal. Can be formatted as a path ("read_xxx...")
        or just the read id ("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").

    Returns
    -------
    Numpy array
        Array representing sampled nanopore current, scaled to pA.

    Raises
    ------
    ValueError
        Raised if f5 is not an h5py.File object.
    """
    if type(f5) is not h5py._hl.files.File:
        raise ValueError("f5 must be an open h5py.File object.")
    raw = get_raw_signal_for_read(f5, read_id, start=start, end=end)
    offset, rng, digi = get_scale_metadata_for_read(f5, read_id)
    return scale_raw_current(raw, offset, rng, digi)


def get_scale_metadata_for_read(f5, read_id):
    """Retrieve scaling values for a specific read in a segmented fast5 file.

    Note: using UK sp. of digitization for consistency w/ file format

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    read_id : str
        Read id to retrieve raw signal. Can be formatted as a path ("read_xxx...")
        or just the read id ("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").

    Returns
    -------
    Tuple
        Offset, range, and digitisation values.

    Raises
    ------
    ValueError
        Raised if f5 is not an h5py.File object.
    """
    if type(f5) is not h5py._hl.files.File:
        raise ValueError("f5 must be an open h5py.File object.")
    if "read" in read_id:
        channel_path = f"{read_id}/channel_id"
    else:
        channel_path = f"read_{read_id}/channel_id"
    attrs = f5.get(channel_path).attrs
    offset = attrs.get("offset")
    rng = attrs.get("range")
    digi = attrs.get("digitisation")
    return offset, rng, digi


def get_raw_signal(f5, channel_no, start=None, end=None):
    """Retrieve raw signal from open fast5 file.

    Optionally, specify the start and end time points in the time series. If no
    values are given for start and/or end, the default is to include data
    starting at the beginning and/or end of the array.

    The signal is not scaled to units of pA; original sampled values are
    returned.

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    channel_no : int
        Channel number for which to retrieve raw signal.
    start : int, optional
        Retrieve a slice of current starting at this time point, by default None
        None will retrieve data starting from the beginning of available data.
    end : int, optional
        Retrieve a slice of current starting at this time point, by default None
        None will retrieve all data at the end of the array.

    Returns
    -------
    Numpy array
        Array representing sampled nanopore current.
    """
    if type(f5) is not h5py._hl.files.File:
        raise ValueError("f5 must be an open h5py.File object.")
    signal_path = f"/Raw/Channel_{channel_no}/Signal"
    if signal_path not in f5:
        raise ValueError(f"Could not find path in fast5 file: {signal_path}, {f5.filename}")
    if start is not None or end is not None:
        raw = f5.get(signal_path)[start:end]
    else:
        raw = f5.get(signal_path)[()]
    return raw


def find_signal_off_regions(raw, window_sz=200, slide=100, current_range=50):
    """Helper function for judge_channels(). Finds regions of current where the
    channel is likely off.

    Parameters
    ----------
    raw : array of floats
        Raw nanopore current (in units of pA).
    window_sz : int, optional
        Sliding window width, by default 200
    slide : int, optional
        How much to slide the window by, by default 100
    current_range : int, optional
        How much the current is allowed to deviate, by default 50

    Returns
    -------
    list of tuples (start, end)
        Start and end points for where the channel is likely off.
    """
    off = []
    for start in range(0, len(raw), slide):
        window_mean = np.mean(raw[start : start + window_sz])
        if window_mean < np.abs(current_range) and window_mean > -np.abs(current_range):
            off.append(True)
        else:
            off.append(False)
    off_locs = np.multiply(np.where(off)[0], slide)
    loc = None
    if len(off_locs) > 0:
        last_loc = off_locs[0]
        start = last_loc
        regions = []
        for loc in off_locs[1:]:
            if loc - last_loc != slide:
                regions.append((start, last_loc))
                start = loc
            last_loc = loc
        if loc is not None:
            regions.append((start, loc))
        return regions
    else:
        return []


def sec_to_obs(time_in_sec, sample_rate_hz):
    """Convert time in seconds to number of observations.

    Parameters
    ----------
    time_in_sec : numeric
        Time in seconds.
    sample_rate_hz : int
        Nanopore sampling rate.

    Returns
    -------
    int
        Number of observations (rounded down if time_in_sec was not an int).
    """
    return int(time_in_sec * sample_rate_hz)


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
        #     median_outside_rng = np.abs(sorted_raw[q_50] - expected_open_channel) > 25
        #     upper_outside_rng = np.abs(sorted_raw[q_75] - expected_open_channel) > 25
        #     if median_outside_rng and upper_outside_rng:
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
