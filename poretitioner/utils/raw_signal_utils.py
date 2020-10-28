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
    signal = get_scaled_raw_for_channel(f5, channel_no, start=start, end=end)
    open_channel = find_open_channel_current(
        signal, open_channel_guess, bound=open_channel_bound
    )
    if open_channel is None:
        # TODO add logging here to give the reason for returning None
        return None
    frac = compute_fractional_blockage(signal, open_channel)
    return frac


def get_local_fractional_blockage(
    f5,
    open_channel_guess=220,
    open_channel_bound=15,
    channel=None,
    local_window_sz=1000,
):
    """Retrieve the scaled raw signal for the channel, compute the open pore
    current, and return the fractional blockage for that channel."""
    signal = get_scaled_raw_for_channel(f5, channel=channel)
    open_channel = find_open_channel_current(
        signal, open_channel_guess, bound=open_channel_bound
    )
    if open_channel is None:
        print("open pore is None")

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
    """
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
    """
    sample_rate = int(f5.get("Meta").attrs["sample_rate"])
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
    """
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
    """
    if "read" in read_id:
        signal_path = f"/{read_id}/Signal"
    else:
        signal_path = f"/read_{read_id}/Signal"
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
    """
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
    """
    if "read" in read_id:
        channel_path = f"{read_id}/channel_id"
    else:
        channel_path = f"read_{read_id}/channel_id"
    print("channel_path", channel_path)
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
    signal_path = f"/Raw/Channel_{channel_no}/Signal"
    if start is not None or end is not None:
        raw = f5.get(signal_path)[start:end]
    else:
        raw = f5.get(signal_path)[()]
    return raw


def get_scale_metadata(f5, channel_no):
    """Retrieve scaling values for bulk fast5 file.

    Note: using UK sp. of digitization for consistency w/ file format

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    channel_no : int
        Channel number for which to retrieve raw signal.

    Returns
    -------
    Tuple
        Offset, range, and digitisation values.
    """
    meta_path = f"/Raw/Channel_{channel_no}/Meta"
    attrs = f5.get(meta_path).attrs
    offset = attrs.get("offset")
    rng = attrs.get("range")
    digi = attrs.get("digitisation")
    return offset, rng, digi


def get_scaled_raw_for_channel(f5, channel_no, start=None, end=None):
    """Retrieve raw signal from open fast5 file, scaled to pA units.

    Note: using UK sp. of digitization for consistency w/ file format

    Optionally, specify the start and end time points in the time series. If no
    values are given for start and/or end, the default is to include data
    starting at the beginning and/or end of the array.

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
        Array representing sampled nanopore current, scaled to pA.
    """
    raw = get_raw_signal(f5, channel_no, start=start, end=end)
    offset, rng, digi = get_scale_metadata(f5, channel_no)
    return scale_raw_current(raw, offset, rng, digi)


def scale_raw_current(raw, offset, rng, digitisation):
    """Scale the raw current to pA units.

    Note: using UK sp. of digitization for consistency w/ file format

    Parameters
    ----------
    raw : Numpy array of numerics
        Array representing directly sampled nanopore current.
    offset : numeric
        Offset value specified in bulk fast5.
    rng : numeric
        Range value specified in bulk fast5.
    digitisation : numeric
        Digitisation value specified in bulk fast5.

    Returns
    -------
    Numpy array of floats
        Raw current scaled to pA.
    """
    return (raw + offset) * (rng / digitisation)


def digitize_raw_current(raw_pA, offset, rng, digitisation):
    """Reverse the scaling from pA to raw current.

    Note: using UK sp. of digitization for consistency w/ file format

    Parameters
    ----------
    raw : Numpy array of numerics
        Array representing nanopore current in units of pA.
    offset : numeric
        Offset value specified in bulk fast5.
    rng : numeric
        Range value specified in bulk fast5.
    digitisation : numeric
        Digitisation value specified in bulk fast5.

    Returns
    -------
    Numpy array of floats
        Raw current digitized.
    """
    return np.array((raw_pA * digitisation / rng) - offset, dtype=np.int16)


def find_open_channel_current(raw, open_channel_guess, bound=None):
    """Compute the median open channel current in the given raw data.

    Inputs presumed to already be in units of pA.

    Parameters
    ----------
    raw : Numpy array
        Array representing sampled nanopore current, scaled to pA.
    open_channel_guess : numeric
        Approximate estimate of the open channel current value.
    bound : numeric, optional
        Approximate estimate of the variance in open channel current value from
        channel to channel (AKA the range to search). If no bound is specified,
        the default is to use 10% of the open_channel guess.

    Returns
    -------
    float
        Median open channel.
    """
    if bound is None:
        bound = 0.1 * open_channel_guess
    upper_bound = open_channel_guess + bound
    lower_bound = open_channel_guess - bound
    ix_in_range = np.where(np.logical_and(raw <= upper_bound, raw > lower_bound))[0]
    if len(ix_in_range) == 0:
        open_channel = None
    else:
        open_channel = np.median(raw[ix_in_range])
    return open_channel


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


def find_segments_below_threshold(time_series, threshold):
    """
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
    list of tuples (start, end)
        Each item in the list represents the (start, end) points of regions
        where the input array drops at or below the threshold.
    """
    diff_points = np.where(
        np.abs(np.diff(np.where(time_series <= threshold, 1, 0))) == 1
    )[0]
    if time_series[0] <= threshold:
        diff_points = np.hstack([[0], diff_points])
    if time_series[-1] <= threshold:
        diff_points = np.hstack([diff_points, [len(time_series)]])
    return list(zip(diff_points[::2], diff_points[1::2]))


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
