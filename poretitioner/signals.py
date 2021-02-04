"""
===================
signals.py
===================

This module encapsulates data related to nanopore signals and the channels that generate them.

"""

from collections import namedtuple
from typing import Optional

import numpy as np
from utils.core import Channel, NumpyArrayLike, Window

__all__ = ["Capture", "FractionalizedSignal", "PicoampereSignal", "RawSignal"]


class BaseSignal(np.ndarray):
    read_id: Optional[str]
    channel: Channel

    def __new__(cls, signal: NumpyArrayLike, channel: Channel, read_id=None):
        """Base class for signals (time series data generated from a nanopore channel).
        Do not use this class directly, instead subclass it.

        You can instances of subclasses of this class exactly like numpy arrays.

        e.g.
        ```
        class MySignal(BaseSignal):
            # ...
            pass

        signal = MySignal(np.array([1, 2, 3]), channel, read_id="oo")

        assert np.median(signal) == 2
        assert np.mean(signal) == 2
        assert all(signal < 10)
        signal_amplified_by_ten = signal + 10
        assert not all(signal_amplified_by_ten < 10))
        ```

        Parameters
        ----------
        signal : NumpyArrayLike
            Numpy array of the signal.
        channel : Channel
            Nanopore channel that generated the data.
        read_id : str, optional
            ReadID that generated this signal, by default None

        Returns
        -------
        [BaseSignal]
            Signal instance.
        """
        obj = np.asarray(signal).view(cls)
        obj.channel = channel
        obj.read_id = read_id
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.channel = getattr(obj, "channel", None)
        self.read_id = getattr(obj, "read_id", None)


class RawSignal(BaseSignal):
    """Raw signals straight from the device's analog to digital converter (ADC).
    You can instances of this class exactly like numpy arrays.

    Parameters
    ----------
    signal : NumpyArrayLike
        Numpy array of the signal.
    channel : Channel
        Nanopore channel that generated the data.
    read_id : str, optional
        ReadID that generated this signal, by default None

    Returns
    -------
    [RawSignal]
        RawSignal instance.
    """

    def to_picoamperes(self):
        """Converts this raw signal to the equivalent signal measured in picoamperes.

        Returns
        -------
        PicoampereSignal
            Signal in picoamperes.
        """
        offset, rng, digitisation = self.channel.calibration
        picoamps = scale_raw_current(self, offset, rng, digitisation)
        picoamperes = PicoampereSignal(picoamps, self.channel, read_id=self.read_id)
        return picoamperes

    def to_fractionalized(self):
        self.to_picoamperes().to_fractionalized()


class PicoampereSignal(BaseSignal):
    """Signal as converted to picoamperes.
    You can instances of this class exactly like numpy arrays.

    In most cases, you won't instantiate this class directly, but rather by `to_picoamperes` from a `RawSignal`
    e.g.
    ```
    nanopore_simulation = np.random.randn(10)
    raw_sig = RawSignal(nanopore_simulation, channel)
    picoamperes = raw_sig.to_picoamperes()
    ```

    Parameters
    ----------
    signal : NumpyArrayLike
        Numpy array of the picoampere signal.
    channel : Channel
        Nanopore channel that generated the data.
    read_id : str, optional
        ReadID that generated this signal, by default None

    Returns
    -------
    [PicoampereSignal]
        PicoampereSignal instance.
    """

    def to_fractionalized(self, open_channel_guess=None):
        """Converts this signal to the equivalent fractionalized signal (i.e. forced to take a value between 1.0 and 0.0)
        Where 0.0 is fully-blocked channel, and 1.0 is a fully open one.

        Parameters
        ----------
        open_channel_guess: float, optional
            Open channel guess to use.

        Returns
        -------
        FractionalizedSignal
            The fractionalized picoampere signal, where 0.0 represents a full-blocked channel, and 1.0 is a fully open one.

        Raises
        ------
        ValueError
            If nothing falls within the channel bounds.
        """
        if open_channel_guess is not None:
            self.channel.open_channel_guess = open_channel_guess

        open_channel_guess = self.channel.open_channel_guess
        open_channel_bound = self.channel.open_channel_bound
        open_channel = find_open_channel_current(
            self, open_channel_guess, bound=open_channel_bound
        )
        if open_channel is None:
            # TODO add logging here to give the reason for returning None
            raise ValueError("No signals found within channel bounds")
        frac = compute_fractional_blockage(self, open_channel)
        normalized = FractionalizedSignal(frac, self.channel, read_id=self.read_id)
        return normalized


class FractionalizedSignal(BaseSignal):
    """A scaled signal that is guaranteed to be between 0 and 1.0.

    Converts a nanopore signal (in units of pA) to fractionalized current
    in the range (0, 1).

    A value of 0 means the pore is fully blocked, and 1 is fully open.


    Parameters
    ----------
    signal : NumpyArrayLike
        Numpy array of the fractionalized signal.
    channel : Channel
        Nanopore channel that generated the data.
    read_id : str, optional
        ReadID that generated this signal, by default None

    Returns
    -------
    [FractionalizedSignal]
        FractionalizedSignal instance.
    """

    @classmethod
    def from_picoamperes(
        self, picoamperes: PicoampereSignal, channel: Channel, read_id: str = None
    ):
        open_channel_guess = channel.open_channel_guess
        open_channel_bound = channel.open_channel_bound
        open_channel = find_open_channel_current(
            picoamperes, open_channel_guess, bound=open_channel_bound
        )
        if open_channel is None:
            # TODO add logging here to give the reason for returning None
            return None
        frac = compute_fractional_blockage(picoamperes, open_channel)
        super().__init__(frac, channel, read_id=read_id)


class Capture(namedtuple("Capture", ["window", "signal"])):
    """Represents a nanopore capture within some window.

    Parameters
    ----------
    window : Window
        The start and end times of this capture.

    signal: BaseSignal
        Signal that generated this capture.
    """

    window: Window
    signal: BaseSignal

    @property
    def ejected(self, end_tol=0):
        ejected = np.abs(self.window.end - self.signal.duration) <= end_tol
        return ejected

    @property
    def duration(self):
        return self.window.duration


###############
#
#   Helpers
#
###############


def scale_raw_current(raw, offset, rng, digitisation):
    """Scale the raw current to pA units.

    Note: using UK spelling of digitization for consistency w/ file format

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


def compute_fractional_blockage(picoamperes, open_channel):
    """Converts a nanopore signal (in units of pA) to fractionalized current
    in the range (0, 1).

    Note that this creates a new buffer.

    A value of 0 means the pore is fully blocked, and 1 is fully open.

    Parameters
    ----------
    picoamperes : array
        Array of nanopore current values in units of pA.
    open_channel : float
        Open channel current value (pA).

    Returns
    -------
    array of floats
        Array of fractionalized nanopore current in the range (0, 1)
    """
    picoamperes = np.frombuffer(picoamperes, dtype=float)
    picoamperes /= open_channel
    frac = np.clip(picoamperes, a_max=1.0, a_min=0.0)
    return frac


def find_open_channel_current(picoamperes, open_channel_guess, bound=None) -> Optional[float]:
    """Compute the median open channel current in the given picoamperes data.

    Inputs presumed to already be in units of pA.

    Parameters
    ----------
    picoamperes : Numpy array
        Array representing sampled nanopore current, scaled to pA.
    open_channel_guess : numeric
        Approximate estimate of the open channel current value.
    bound : numeric, optional
        Approximate estimate of the variance in open channel current value from
        channel to channel (AKA the range to search). If no bound is specified,
        the default is to use 10% of the open_channel guess.

    Returns
    -------
    Optional[float]
        Median open channel, or None if no signals were within the bounds.
    """
    if bound is None:
        bound = 0.1 * open_channel_guess
    upper_bound = open_channel_guess + bound
    lower_bound = open_channel_guess - bound
    ix_in_range = np.where(np.logical_and(picoamperes <= upper_bound, picoamperes > lower_bound))[
        0
    ]
    if len(ix_in_range) == 0:
        open_channel = None
    else:
        open_channel = np.median(picoamperes[ix_in_range])
    return open_channel


def find_signal_off_regions(
    picoamperes, window_sz=200, slide=100, current_range=50
) -> List[Window]:
    """Helper function for judge_channels(). Finds regions of current where the
    channel is likely off.

    Parameters
    ----------
    picoamperes : array of floats
        Raw nanopore current (in units of pA).
    window_sz : int, optional
        Sliding window width, by default 200
    slide : int, optional
        How much to slide the window by, by default 100
    current_range : int, optional
        How much the current is allowed to deviate, by default 50

    Returns
    -------
    list of Windows (start, end)
        Start and end points for where the channel is likely off.
    """
    off = []
    for start in range(0, len(picoamperes), slide):
        window_mean = np.mean(picoamperes[start : start + window_sz])
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
    list of Captures (start, end)
        Each item in the list represents the (start, end) points of regions
        where the input array drops at or below the threshold.
    """
    diff_points = np.where(np.abs(np.diff(np.where(time_series <= threshold, 1, 0))) == 1)[0]
    if time_series[0] <= threshold:
        diff_points = np.hstack([[0], diff_points])
    if time_series[-1] <= threshold:
        diff_points = np.hstack([diff_points, [len(time_series)]])
    return list(zip(diff_points[::2], diff_points[1::2]))
