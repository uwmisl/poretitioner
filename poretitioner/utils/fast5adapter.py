"""
===================
fast5adapter.py
===================

General adapter for dealing with Fast5 files.

TODO: Decide with Katie where the Fast5 specification should be stored.
"""

import numpy as np
from signals import FractionalizedSignal, RawSignal
from utils.core import Channel, ChannelCalibration

__all__ = [
    "get_channel_calibration",
    "get_channel_calibration_for_read",
    "get_fractional_blockage",
    "get_fractional_blockage_for_read",
    "get_local_fractional_blockage",
    "get_raw_signal",
    "get_raw_signal_for_read",
    "get_sampling_rate",
    "get_voltage",
]


def get_channel_calibration(f5, channel_no) -> ChannelCalibration:
    """Retrieve channel calibration for bulk fast5 file.

    Note: using UK spelling of digitization for consistency w/ file format

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

    calibration = ChannelCalibration(offset=offset, range=rng, digitisation=digi)
    return calibration


def get_channel_calibration_for_read(f5, read_id) -> ChannelCalibration:
    """Retrieve the channel calibration for a specific read in a segmented fast5 file.
    This is used for properly scaling values when converting raw signal to actual units.

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
    attrs = f5.get(channel_path).attrs
    offset = attrs.get("offset")
    rng = attrs.get("range")
    digi = attrs.get("digitisation")
    calibration = ChannelCalibration(offset=offset, range=rng, digitisation=digi)
    return calibration


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
    frac = (
        get_raw_signal(f5, channel_no, start=start, end=end).to_picoamperes().to_fractionalized()
    )
    return frac


def get_local_fractional_blockage(
    f5, open_channel_guess=220, open_channel_bound=15, channel_no=None, local_window_sz=1000
):
    """Retrieve the scaled raw signal for the channel, compute the open pore
    current, and return the fractional blockage for that channel."""
    signal = get_raw_signal(
        f5,
        channel_no,
        open_channel_guess=open_channel_guess,
        open_channel_bound=open_channel_bound,
    )
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


def get_fractional_blockage_for_read(f5, read_id, start=None, end=None) -> FractionalizedSignal:
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

    if "read" in read_id:
        channel_path = f"{read_id}/channel_id"
    else:
        channel_path = f"read_{read_id}/channel_id"

    open_channel = f5.get(channel_path).attrs["open_channel_pA"]
    signal = get_raw_signal_for_read(
        f5, read_id, open_channel_guess=open_channel, start=start, end=end
    )
    frac = signal.to_fractionalized()
    return frac


def get_raw_signal_for_read(
    f5, read_id, start=None, end=None, open_channel_guess=220, open_channel_bound=15
) -> RawSignal:
    """Retrieve raw signal from open fast5 file for the specified read_id.

    Parameters
    ----------
    f5 : h5py.File
        Fast5 file, open for reading using h5py.File.
    read_id : str
        Read id to retrieve raw signal. Can be formatted as a path ("read_xxx...")
        or just the read id ("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").
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
    Numpy array
        Array representing sampled nanopore current.
    """
    if "read" in read_id:
        signal_path = f"/{read_id}/Signal"
    else:
        signal_path = f"/read_{read_id}/Signal"
    if signal_path in f5:
        calibration = get_channel_calibration_for_read(f5, read_id)
        channel = Channel(
            calibration,
            open_channel_guess=open_channel_guess,
            open_channel_bound=open_channel_bound,
        )

        raw = f5.get(signal_path)[start:end]
        raw = RawSignal(raw, channel, read_id=read_id)
        return raw
    else:
        raise ValueError(f"Path {signal_path} not in fast5 file.")


def get_raw_signal(
    f5, channel_no, start=None, end=None, open_channel_guess=220, open_channel_bound=15
) -> RawSignal:
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
    open_channel_guess : int, optional
        Approximate estimate of the open channel current value, by default 220
    open_channel_bound : int, optional
        Approximate estimate of the variance in open channel current value from
        channel to channel (AKA the range to search), by default 15

    Returns
    -------
    Numpy array
        Array representing sampled nanopore current.
    """
    signal_path = f"/Raw/Channel_{channel_no}/Signal"
    calibration = get_channel_calibration(f5, channel_no)
    channel = Channel(
        calibration, open_channel_guess=open_channel_guess, open_channel_bound=open_channel_bound
    )

    if start is not None or end is not None:
        signal = f5.get(signal_path)[start:end]
    else:
        signal = f5.get(signal_path)[()]

    raw = RawSignal(signal, channel)
    return raw
