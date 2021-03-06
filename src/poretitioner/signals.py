"""
===================
signals.py
===================

This module encapsulates data related to nanopore signals and the channels that generate them.

"""
from __future__ import annotations

import dataclasses
from collections import namedtuple
from dataclasses import dataclass
from typing import *  # I know people don't like import *, but I think it has benefits for types (doesn't impede people from being generous with typing)

import h5py
import numpy as np

from . import logger
from .hdf5 import (
    HDF5_Attributes,
    HDF5_Dataset,
    HDF5_DatasetSerialableDataclass,
    HDF5_Group,
    HDF5_GroupSerialableDataclass,
    HDF5_GroupSerializable,
    HDF5_Type,
)
from .utils.core import NumpyArrayLike, ReadId, Window

__all__ = [
    "digitize_current",
    "find_open_channel_current",
    "find_segments_below_threshold",
    "Capture",
    "ChannelCalibration",
    "FractionalizedSignal",
    "PicoampereSignal",
    "RawSignal",
    "VoltageSignal",
    "SignalMetadata",
    "DEFAULT_OPEN_CHANNEL_GUESS",
    "DEFAULT_OPEN_CHANNEL_BOUND",
]

DEFAULT_OPEN_CHANNEL_GUESS = 220  # In picoAmperes (pA).
DEFAULT_OPEN_CHANNEL_BOUND = 15  # In picoAmperes (pA).


class SignalMetadata(namedtuple("SignalMetadata", ["channel_number", "window", "calibration"])):
    """Metadata around a signal."""


# @dataclass(frozen=True)
# class BaseSignalSerializationInfo:
#     """Extra  (i.e. non ndarray) data that we want to preserve during serialization.
#     To understand why this is necessary, please read: https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array

#     Fields
#      ----------
#     channel_number : int
#         Which nanopore channel generated the data. Must be 1 or greater.
#     calibration: ChannelCalibration
#         How to convert the nanopore device's analog-to-digital converter (ADC) raw signal to picoAmperes of current.
#     read_id : ReadId, optional
#         ReadID that generated this signal, by default None
#     """

#     channel_number: int
#     calibration: ChannelCalibration
#     read_id: Optional[ReadId]

#     # Multiprocessing and Dask require pickling (i.e. serializing) their inputs.
#     # By default, this will drop all our custom class data.
#     # https://stackoverflow.com/questions/26598109/preserve-custom-attributes-when-pickling-subclass-of-numpy-array
#     def __reduce__(self):
#         reconstruct, arguments, object_state = super().__reduce__()
#         # Create a custom state to pass to __setstate__ when this object is deserialized.
#         info = self.serialize_info()
#         new_state = object_state + (info,)
#         # Return a tuple that replaces the parent's __setstate__ tuple with our own
#         return (reconstruct, arguments, new_state)

#     def __setstate__(self, state):
#         info = state[-1]
#         self.deserialize_from_info(info)
#         # Call the parent's __setstate__ with the other tuple elements.
#         super().__setstate__(state[0:-1])


class BaseSignal(NumpyArrayLike):
    """Base class for signals (time series data generated from a nanopore channel).
    Do not use this class directly, instead subclass it.

    Subclasses must implement:
    `def __array_finalize__(self, obj):` and `def __reduce__(self):`. See `FractionalizedSignal` for an example.

    You can instances of subclasses of this class exactly like numpy arrays.

    Parameters
    ----------
    signal : NumpyArrayLike
        Numpy array of the signal.
    channel_number : int
        Which nanopore channel generated the data. Must be 1 or greater.
    calibration: ChannelCalibration
        How to convert the nanopore device's analog-to-digital converter (ADC) raw signal to picoAmperes of current.
    read_id : ReadId, optional
        ReadID that generated this signal, by default None
    Returns
    -------
    [BaseSignal]
        Signal instance.

    Notes
    -------

    Here are some examples about how this classes subclasses can be used exactly like numpy arrays:

    ```
    class MySignal(BaseSignal):
        # ...
    # Later...
    signal = MySignal(np.array([1, 2, 3]), channel_number, calibration, read_id="oo")

    assert np.median(signal) == 2 # We can calculuate the median, just like a numpy array.
    assert np.mean(signal) == 2 # We can calculuate the median, just like a numpy array.
    assert all(signal < 10) # We can do filtering...
    assert signal[0:-1] # And even slicing!
    ```
    """

    def __new__(cls, signal: NumpyArrayLike):
        obj = np.copy(signal).view(
            cls
        )  # Optimization: Consider not making a copy, this is more error prone though: np.asarray(signal).view(cls)
        return obj

    @property
    def duration(self) -> int:
        """How many samples are contained in this signal

        Returns
        -------
        int
            Number of samples in this signal.
        """
        return len(self)


class VoltageSignal(BaseSignal):
    pass


class CurrentSignal(BaseSignal):
    """Base class for electrical current signals (i.e. time series data generated from a nanopore channel).
    Do not use this class directly, instead subclass it.

    Fields
    ----------
    signal : NumpyArrayLike
        Numpy array of the signal.
    channel_number : int
        Which nanopore channel generated the data. Must be 1 or greater.
    calibration: ChannelCalibration
        How to convert the nanopore device's analog-to-digital converter (ADC) raw signal to picoAmperes of current.
    read_id : ReadId, optional
        ReadID that generated this signal, by default None

    Returns
    -------
    [CurrentSignal]
        CurrentSignal instance.
    """

    # ReadId that generated this signal, optional.
    read_id: Optional[ReadId]

    # Index of the channel that generated this signal.
    channel_number: int

    # Device calibration.
    calibration: ChannelCalibration

    def __new__(
        cls,
        signal: NumpyArrayLike,
        channel_number: int,
        calibration: ChannelCalibration,
        read_id=None,
    ):
        if channel_number < 1:
            raise ValueError(
                f"Channel number '{channel_number}' is invalid. Channel indicies are 1-based, as opposed to 0-based. Please make sure the channel index is greater than zero."
            )
        obj = np.copy(signal).view(
            cls
        )  # Optimization: Consider not making a copy, this is more error prone though: np.asarray(signal).view(cls)
        # If the signal already has channel info, just use that instead of the passed in values.
        obj.channel_number = channel_number
        obj.calibration = calibration
        obj.read_id = read_id
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.channel_number = getattr(obj, "channel_number", None)
        self.calibration = getattr(obj, "calibration", None)
        self.read_id = getattr(obj, "read_id", None)


class FractionalizedSignal(CurrentSignal):
    """A scaled signal that is guaranteed to be between 0 and 1.0.

    Converts a nanopore signal (in units of pA) to fractionalized current
    in the range (0, 1).

    A value of 0 means the pore is fully blocked, and 1 is fully open.

    Fields
    ----------
    signal : NumpyArrayLike
        Numpy array of the fractionalized signal.
    channel_number : int
        Which nanopore channel generated the data. Must be 1 or greater.
    calibration: ChannelCalibration
        How to convert the nanopore device's analog-to-digital converter (ADC) raw signal to picoAmperes of current.
    open_channel_pA : float
        Median signal value when no capture is in the pore.
    read_id : ReadId, optional
        ReadID that generated this signal, by default None
    do_conversion: bool, optional
        Whether to interpret the input signal as a RawSignal that needs to be fractionalized, otherwise assume the input
        has already been fractionalized.

    Returns
    -------
    [FractionalizedSignal]
        FractionalizedSignal instance.
    """

    def __new__(
        cls,
        signal: NumpyArrayLike,
        channel_number: int,
        calibration: ChannelCalibration,
        open_channel_pA: float,
        read_id: ReadId = None,
        do_conversion: bool = False,
    ):
        if do_conversion:
            pico = calculate_raw_in_picoamperes(signal, calibration)
            frac = compute_fractional_blockage(pico, open_channel_pA)
            # Now use the fractionalized version as the signal.
            signal = frac

        obj = super().__new__(cls, signal, channel_number, calibration, read_id=read_id)
        obj.open_channel_pA = open_channel_pA
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        super().__array_finalize__(obj)
        self.open_channel_pA = getattr(obj, "open_channel_pA", None)

    def __reduce__(self):
        reconstruct, arguments, object_state = super(FractionalizedSignal, self).__reduce__()
        # Create a custom state to pass to __setstate__ when this object is deserialized.
        info = self.serialize_info(open_channel_pA=self.open_channel_pA)
        new_state = object_state + (info,)
        # Return a tuple that replaces the parent's __setstate__ tuple with our own
        return (reconstruct, arguments, new_state)

    def __setstate__(self, state):
        info = state[-1]
        self.deserialize_from_info(info)
        # Call the parent's __setstate__ with the other tuple elements.
        super(FractionalizedSignal, self).__setstate__(state[0:-1])

    def to_picoamperes(self) -> PicoampereSignal:
        """Converts this raw signal to the equivalent signal measured in picoamperes.

        Returns
        -------
        PicoampereSignal
            Signal in picoamperes.
        """
        picoamps = compute_picoamperes_from_fractional_blockage(self)
        picoamperes = PicoampereSignal(
            picoamps, self.channel_number, self.calibration, read_id=self.read_id
        )
        return picoamperes


class RawSignal(CurrentSignal):
    """Raw signals straight from the device's analog to digital converter (ADC).
    You can instances of this class exactly like numpy arrays.

    Fields
    ----------
    signal : NumpyArrayLike
        Numpy array of the signal.
    channel_number : int
        Which nanopore channel generated the data. Must be 1 or greater.
    calibration: ChannelCalibration
        How to convert the nanopore device's analog-to-digital converter (ADC) raw signal to picoAmperes of current.
    read_id : ReadId, optional
        ReadID that generated this signal, by default None

    Returns
    -------
    [RawSignal]
        RawSignal instance.
    """

    def to_picoamperes(self) -> PicoampereSignal:
        """Converts this raw signal to the equivalent signal measured in picoamperes.

        Returns
        -------
        PicoampereSignal
            Signal in picoamperes.
        """
        picoamps = calculate_raw_in_picoamperes(self, self.calibration)
        picoamperes = PicoampereSignal(
            picoamps, self.channel_number, self.calibration, read_id=self.read_id
        )
        return picoamperes

    def to_fractionalized_from_guess(
        self,
        open_channel_guess=DEFAULT_OPEN_CHANNEL_GUESS,
        open_channel_bound=None,
        default=DEFAULT_OPEN_CHANNEL_GUESS,
    ) -> FractionalizedSignal:
        return self.to_picoamperes().to_fractionalized_from_guess(
            open_channel_guess=open_channel_guess,
            open_channel_bound=open_channel_bound,
            default=default,
        )


class PicoampereSignal(CurrentSignal):
    """Signal as converted to picoamperes.
    You can instances of this class exactly like numpy arrays.

    In most cases, you won't instantiate this class directly, but rather by calling `to_picoamperes()` from a `RawSignal`.
    e.g.
    ```
    nanopore_simulation = np.random.randn(10)
    raw_sig = RawSignal(nanopore_simulation, channel)
    picoamperes: PicoampereSignal = raw_sig.to_picoamperes()
    ```

    Fields
    ----------
    signal : NumpyArrayLike
        Numpy array of the picoampere signal.
    channel_number : int
        Which nanopore channel generated the data. Must be 1 or greater.
    calibration: ChannelCalibration
        How to convert the nanopore device's analog-to-digital converter (ADC) raw signal to picoAmperes of current.
    read_id : ReadId, optional
        ReadID that generated this signal, by default None

    Returns
    -------
    [PicoampereSignal]
        PicoampereSignal instance.
    """

    def find_open_channel_current(
        self,
        open_channel_guess=DEFAULT_OPEN_CHANNEL_GUESS,
        open_channel_bound=None,
        default=DEFAULT_OPEN_CHANNEL_GUESS,
    ) -> float:
        """Creates a channel summary via the signal.

        Parameters
        ----------
        open_channel_guess : float, optional
            Approximate estimate of the "open channel current", which is the median current that flows when the channel is completely open (i.e. unblocked). Measured in picoamperes, by default DEFAULT_OPEN_CHANNEL_GUESS.
        open_channel_bound : float, optional
            Approximate estimate of the variance in open channel current value from
            channel to channel (AKA the range to search), by default DEFAULT_OPEN_CHANNEL_BOUND.
        open_channel_default : Optional[float], optional
            Default open channel current to use if one could not be calculated from the signal, by default DEFAULT_OPEN_CHANNEL_GUESS.

        """
        open_channel_pA = find_open_channel_current(
            self,
            open_channel_guess=open_channel_guess,
            open_channel_bound=open_channel_bound,
            default=default,
        )
        return open_channel_pA

    def to_raw(self) -> RawSignal:
        """Digitize the picoampere signal, converting it back to raw ADC values.

        Returns
        -------
        RawSignal
            The raw signal that generated this current.
        """
        channel_number = self.channel_number
        calibration = self.calibration
        read_id = self.read_id

        raw = RawSignal(
            digitize_current(self, calibration),
            channel_number,
            calibration,
            read_id=read_id,
        )

        return raw

    def digitize(self) -> RawSignal:
        """
        Digitize the picoampere signal, converting it back to raw ADC values.

        This method is just syntatic sugar for `to_raw()`

        Returns
        -------
        RawSignal
            Digitized signal that generated this current.
        """
        return self.to_raw()

    def to_fractionalized(self, open_channel_pA: float) -> FractionalizedSignal:
        """Gets the fractionalized signal from a known open channel current.

        Parameters
        ----------
        open_channel_pA : float
            Open channel current in picoamperes.

        Returns
        -------
        FractionalizedSignal
            Fractionalized signal from the picoampere signal.
        """
        calibration = self.calibration
        channel_number = self.channel_number
        frac = compute_fractional_blockage(self, open_channel_pA)
        fractionalized = FractionalizedSignal(
            frac, channel_number, calibration, open_channel_pA, read_id=self.read_id
        )
        return fractionalized

    def to_fractionalized_from_guess(
        self,
        open_channel_guess=DEFAULT_OPEN_CHANNEL_GUESS,
        open_channel_bound=None,
        default=DEFAULT_OPEN_CHANNEL_GUESS,
    ) -> FractionalizedSignal:
        """Converts this signal to the equivalent fractionalized signal (i.e. takes a value between 1.0 and 0.0, where 0.0 is fully-blocked channel, and 1.0 is a fully open one).

        Parameters
        ----------
        open_channel_guess : float
            Approximate estimate of the "open channel current", which is the median current that flows when the channel is completely open (i.e. unblocked). Measured in picoamperes, by default DEFAULT_OPEN_CHANNEL_GUESS.
        open_channel_bound : float
            Approximate estimate of the variance in open channel current value from
            channel to channel (AKA the range to search), by default DEFAULT_OPEN_CHANNEL_BOUND.
        open_channel_default : Optional[float], optional
            Default open channel current to use if one could not be calculated from the signal, by default DEFAULT_OPEN_CHANNEL_GUESS.

        Returns
        -------
        FractionalizedSignal
            The fractionalized picoampere signal, where 0.0 represents a full-blocked channel, and 1.0 is a fully open one.

        Raises
        ------
        ValueError
            If nothing falls within the channel bounds.
        """
        calibration = self.calibration
        channel_number = self.channel_number
        open_channel_pA = self.find_open_channel_current(
            open_channel_guess=open_channel_guess,
            open_channel_bound=open_channel_bound,
            default=DEFAULT_OPEN_CHANNEL_GUESS,
        )

        frac = compute_fractional_blockage(self, open_channel_pA)
        fractionalized = FractionalizedSignal(
            frac, channel_number, calibration, open_channel_pA, read_id=self.read_id
        )
        return fractionalized


# @dataclass(frozen=True, init=False)
@dataclass(init=False)
class HDF5_Signal(HDF5_DatasetSerialableDataclass):
    start_time_bulk: int
    start_time_local: int
    duration: int
    voltage: float
    open_channel_pA: float
    read_id: ReadId
    ejected: bool
    channel_number: int

    def __new__(
        cls,
        data: Union[np.ndarray, NumpyArrayLike],
        start_time_bulk: int,
        start_time_local: int,
        duration: int,
        voltage: float,
        open_channel_pA: float,
        read_id: ReadId,
        ejected: bool,
        channel_number: int,
    ):
        self = super().__new__(cls, data)
        self.start_time_bulk = start_time_bulk
        self.start_time_local = start_time_local
        self.duration = duration
        self.voltage = voltage
        self.open_channel_pA = open_channel_pA
        self.read_id = read_id
        self.ejected = ejected
        self.channel_number = channel_number
        return self

    def __init__(
        self,
        data: Union[np.ndarray, NumpyArrayLike],
        start_time_bulk: int,
        start_time_local: int,
        duration: int,
        voltage: float,
        open_channel_pA: float,
        read_id: ReadId,
        ejected: bool,
        channel_number: int,
    ):
        super().__init__(data)
        self.start_time_bulk = start_time_bulk
        self.start_time_local = start_time_local
        self.duration = duration
        self.voltage = voltage
        self.open_channel_pA = open_channel_pA
        self.read_id = read_id
        self.ejected = ejected
        self.channel_number = channel_number


@dataclass(frozen=True)
class Channel(HDF5_GroupSerialableDataclass):
    """A nanopore channel. Contains an id (channel_number), info on how it was calibrated, and its median current when the pore is open.

    Fields
    ----------
    channel_number : int
        Identifies which pore channel this is. Note that indicies are 1-based, not 0-based (i.e. the first channel is 1).
    calibration : ChannelCalibration
        How this channel was calibrated. This determines how to convert the raw ADC signal to picoamperes.
    open_channel_pA : float
        The median current that flows through the channel when it is completely open (i.e. unblocked). Measured in picoamperes.
    sampling_rate: int
        How frequently the device produces observations. In Hz.
    """

    calibration: ChannelCalibration
    channel_number: int
    open_channel_pA: PicoampereSignal
    sampling_rate: int

    @classmethod
    def from_raw_signal(
        cls,
        raw: RawSignal,
        channel_number: int,
        calibration: ChannelCalibration,
        sampling_rate: int,
        open_channel_guess: float = DEFAULT_OPEN_CHANNEL_GUESS,
        open_channel_bound: float = DEFAULT_OPEN_CHANNEL_BOUND,
        open_channel_default: float = DEFAULT_OPEN_CHANNEL_GUESS,
    ):
        """Creates a channel summary via the signal.

        Parameters
        ----------
        raw : RawSignal
            The ADC signal generated from this channel, unitless.
        channel_number : int
            Identifies which pore channel this is. Note that indicies are 1-based, not 0-based (i.e. the first channel is 1).
        calibration : ChannelCalibration
            How this channel was calibrated. This determines how to convert the raw ADC signal to picoamperes.
        sampling_rate: int
            How frequently the channel produced observations. In Hz.
        open_channel_guess : float
            Approximate estimate of the "open channel current", which is the median current that flows when the channel is completely open (i.e. unblocked). Measured in picoamperes, by default DEFAULT_OPEN_CHANNEL_GUESS.
        open_channel_bound : float
            Approximate estimate of the variance in open channel current value from
            channel to channel (AKA the range to search), by default DEFAULT_OPEN_CHANNEL_BOUND.
        open_channel_default : float, optional
            Default open channel current to use if one could not be calculated from the signal, by default DEFAULT_OPEN_CHANNEL_GUESS.

        Returns
        -------
        Channel
            Channel as derived from the signal.
        """
        picoamperes = raw.to_picoamperes()
        self = cls.from_picoampere_signal(
            picoamperes,
            channel_number,
            calibration,
            sampling_rate,
            open_channel_guess=open_channel_guess,
            open_channel_bound=open_channel_bound,
            open_channel_default=open_channel_default,
        )
        return self

    @classmethod
    def from_picoampere_signal(
        cls,
        picoamperes: PicoampereSignal,
        channel_number: int,
        calibration: ChannelCalibration,
        sampling_rate: int,
        open_channel_guess: float = DEFAULT_OPEN_CHANNEL_GUESS,
        open_channel_bound: float = DEFAULT_OPEN_CHANNEL_BOUND,
        open_channel_default: float = DEFAULT_OPEN_CHANNEL_GUESS,
    ):
        """Creates a channel summary via the signal.

        Parameters
        ----------
        picoamperes : PicoampereSignal
            The signal generated from this channel, in picoamperes.
        channel_number : int
            Identifies which pore channel this is. Note that indicies are 1-based, not 0-based (i.e. the first channel is 1).
        calibration : ChannelCalibration
            How this channel was calibrated. This determines how to convert the raw ADC signal to picoamperes.
        sampling_rate : int
            How frequently the channel produced observations. In Hz.
        open_channel_guess : float
            Approximate estimate of the "open channel current", which is the median current that flows when the channel is completely open (i.e. unblocked). Measured in picoamperes, by default DEFAULT_OPEN_CHANNEL_GUESS.
        open_channel_bound : float
            Approximate estimate of the variance in open channel current value from
            channel to channel (AKA the range to search), by default DEFAULT_OPEN_CHANNEL_BOUND.
        open_channel_default : float, optional
            Default open channel current to use if one could not be calculated from the signal, by default DEFAULT_OPEN_CHANNEL_GUESS.

        Returns
        -------
        Channel
            Channel as derived from the signal.
        """

        open_channel_pA = picoamperes.find_open_channel_current(
            open_channel_guess, open_channel_bound, default=open_channel_default
        )
        self = cls.__new__(cls)
        # We're using this esoteric __setattr__ method so we can keep the dataclass frozen while setting its initial attributes
        # https://docs.python.org/3/library/dataclasses.html#frozen-instances
        object.__setattr__(self, "open_channel_pA", open_channel_pA)
        object.__setattr__(self, "channel_number", channel_number)
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "sampling_rate", sampling_rate)
        return self

    @classmethod
    def from_group(cls, group: h5py.Group) -> HDF5_GroupSerializable:
        """Serializes this object FROM an HDF5 Group.

        class Baz(HDF5_GroupSerializable):
            # ...Implementation

        my_hdf5_file = h5py.File("/path/to/file")
        baz_serialized_group = filts.require_group("/baz")

        baz = Baz.from_group(baz_serialized_group) # I now have an instance of Baz.
        """
        ...


@dataclass(frozen=True)
class CaptureMetadata:
    """Metadata associated with a capture candidate.

    Fields
    ----------

    read_id : ReadId
        Identifier for this read (unique within the run, usually a uuid.)
    start_time_bulk : int
        Starting time of the capture relative to the start of the bulk fast5
        file.
    start_time_local : int
        Starting time of the capture relative to the beginning of the segmented
        region. (Relative to sub_run_start_observations in parallel_find_captures().)
    duration : int
        Number of data points in the capture.
    ejected : boolean
        Whether or not the capture was ejected from the pore.
    voltage_threshold : int
        Voltage at which the capture occurs (single value for entire window).
    open_channel_pA : float
        Median signal value when no capture is in the pore.
    channel_number : int
        Nanopore channel number (in MinION, value is between 1-512).
    calibration: ChannelCalibration
        Details on how this channel's analog-to-digital converter (ADC) was calibrated.
    sampling_rate : float
        Nanopore sampling rate (Hz)
    """

    read_id: ReadId
    start_time_bulk: int
    start_time_local: int
    duration: int
    ejected: bool
    voltage_threshold: int
    open_channel_pA: float
    channel_number: int
    calibration: ChannelCalibration
    sampling_rate: float


@dataclass(frozen=True)
class Capture:
    """Represents a nanopore capture within some window.

    Fields
    ----------

    signal: PicoampereSignal
        Signal that generated this capture, in picoAmperes.

    window : Window
        The start and end times of this capture.

    signal_threshold_frac: float
        Threshold for the first pass of finding captures (in fractional
        current). (Captures are <= the threshold.).

    open_channel_pA_calculated : float
        Calculated open channel current value.

    ejected: bool
        Whether or not the capture was ejected from the pore.
    """

    signal: PicoampereSignal
    # channel_number: int
    window: Window
    signal_threshold_frac: float
    open_channel_pA_calculated: float
    ejected: bool

    @property
    def duration(self):
        return len(self.signal)

    def fractionalized(self, start=None, end=None) -> FractionalizedSignal:
        frac = self.signal[start:end].to_fractionalized(self.open_channel_pA_calculated)
        return frac

    def __repr__(self):
        string = f"duration: {self.signal.duration} channel_number:{self.signal.channel_number} window:{self.window!s} open_channel_pA:{self.open_channel_pA_calculated} ejected:{self.ejected}"
        return string

    def __len__(self):
        return self.duration


###############
#
#   Helpers
#
###############


def calculate_raw_in_picoamperes(
    raw: NumpyArrayLike, calibration: ChannelCalibration
) -> NumpyArrayLike:
    """Converts a raw analog-to-digital converter (ADC) signal to
    picoamperes of current.

    Parameters
    ----------
    raw : NumpyArrayLike
        Array representing directly sampled nanopore current.
    calibration: ChannelCalibration
        How this device's ADC is calibrated.

    Returns
    -------
    [NumpyArrayLike]
        Raw current scaled to pA.
    """
    return (raw + calibration.offset) * (calibration.range / calibration.digitisation)


def digitize_current(
    picoamperes: NumpyArrayLike, calibration: ChannelCalibration
) -> NumpyArrayLike:
    """Reverse the scaling from picoamperes to raw current.
    Note: using UK spelling of digitization for consistency w/ file format

    Parameters
    ----------
    picoamperes : NumpyArrayLike
        Array representing nanopore current in units of pA.
    calibration : ChannelCalibration
        How the ADC converted the signal to picoamperes.
    Returns
    -------
    [NumpyArrayLike]
        Picoampere current digitized (i.e. the raw ADC values).
    """
    return np.array(
        (picoamperes * calibration.digitisation / calibration.range) - calibration.offset,
        dtype=np.int16,
    )


def compute_fractional_blockage(
    picoamperes: NumpyArrayLike, open_channel: float
) -> NumpyArrayLike:
    """Converts a nanopore signal (in units of pA) to fractionalized current
    in the range (0, 1).

    A value of 0 means the pore is fully blocked, and 1 is fully open.

    Parameters
    ----------
    picoamperes : NumpyArrayLike
        Array of nanopore current values in units of pA.
    open_channel : float
        Open channel current value (pA).

    Returns
    -------
    [NumpyArrayLike]
        Array of fractionalized nanopore current in the range [0, 1]
    """
    picoamperes /= open_channel
    frac = np.clip(picoamperes, a_max=1.0, a_min=0.0)
    return frac


def compute_picoamperes_from_fractional_blockage(
    fractional: FractionalizedSignal,
) -> NumpyArrayLike:
    """Converts a fractional signal  in the range (0, 1) to Picoampere current

    Since the fraction is determined by dividing picoamperes by the open_channel current, then clipping it to [0, 1] range.
    This won't return exact values of course, since we clipped it, but it'll give an idea.

    Note that this creates a new buffer.

    Parameters
    ----------
    fractional : FractionalizedSignal
        Fractionalized nanopore current values.

    Returns
    -------
    [NumpyArrayLike]
        Array of fractionalized nanopore current in the range [0, 1]
    """

    pico = np.frombuffer(fractional * fractional.open_channel_pA, dtype=float)
    return pico


def find_open_channel_current(
    picoamperes: PicoampereSignal,
    open_channel_guess: float = DEFAULT_OPEN_CHANNEL_GUESS,
    open_channel_bound: float = None,
    default: float = DEFAULT_OPEN_CHANNEL_GUESS,
    log: logger.Logger = logger.getLogger(),
) -> float:
    """Compute the median open channel current in the given picoamperes data.

    Inputs presumed to already be in units of pA.

    Parameters
    ----------
    picoamperes : NumpyArrayLike
        Array representing sampled nanopore current, scaled to pA.
    open_channel_guess : numeric, optional
        Approximate estimate of the open channel current value, by default DEFAULT_OPEN_CHANNEL_GUESS.
    bound : numeric, optional
        Approximate estimate of the variance in open channel current value from
        channel to channel (AKA the range to search). If no bound is specified,
        the default is to use 10% of the open_channel guess.
    default : numeric, optional
        Default open channel current value to use if one could not be calculated, by default DEFAULT_OPEN_CHANNEL_GUESS.
    log: logger.Logger, optional
        Logger to use, defaults to singleton logger.
    Returns
    -------
    PicoampereSignal
        Median open channel, or DEFAULT_OPEN_CHANNEL_GUESS if no signals were within the bounds and no default was provided.
    """
    if open_channel_bound is None:
        open_channel_bound = 0.1 * open_channel_guess
    upper_bound = open_channel_guess + open_channel_bound
    lower_bound = open_channel_guess - open_channel_bound
    ix_in_range = np.where(np.logical_and(picoamperes <= upper_bound, picoamperes > lower_bound))[
        0
    ]
    if len(ix_in_range) == 0:
        channel_number = getattr(picoamperes, "channel_number", "N/A")
        log.info(
            f"Couldn't calculate the median open channel current for channel #{channel_number}, using open channel guess ({open_channel_guess}), bound ({open_channel_bound}), and default ({default}). Falling back to default ({default})"
        )
        open_channel = default
    else:
        open_channel = np.median(picoamperes[ix_in_range])[()]
    return float(open_channel)


def find_signal_off_regions(
    picoamperes: PicoampereSignal, window_sz=200, slide=100, current_range=50
) -> List[Window]:
    """Helper function for judge_channels(). Finds regions of current where the
    channel is likely off.

    Parameters
    ----------
    picoamperes : PicoampereSignal
        Nanopore current (in units of pA).
    window_sz : int, optional
        Sliding window width, by default 200
    slide : int, optional
        How much to slide the window by, by default 100
    current_range : int, optional
        How much the current is allowed to deviate, by default 50

    Returns
    -------
    List[Window]
        List of Windows (start, end).
        Start and end points for where the channel is likely off.
    """
    off = []
    for start in range(0, len(picoamperes), slide):
        window_mean = np.mean(picoamperes[start : start + window_sz])
        if -np.abs(current_range) < window_mean and window_mean < np.abs(current_range):
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


def find_segments_below_threshold(time_series: NumpyArrayLike, threshold: float) -> List[Window]:
    """
    Find regions where the time series data points drop at or below the
    specified threshold.

    Parameters
    ----------
    time_series : NumpyArrayLike
        Array containing time series data points.
    threshold : float
        Find regions where the time series drops at or below this value

    Returns
    -------
    List[Window]
        List of capture windows (start, end).
        Each window in the list represents the (start, end) points of regions
        where the input array drops at or below the threshold.
    """
    diff_points = np.where(np.abs(np.diff(np.where(time_series <= threshold, 1, 0))) == 1)[0]
    if time_series[0] <= threshold:
        diff_points = np.hstack([[0], diff_points])
    if time_series[-1] <= threshold:
        diff_points = np.hstack([diff_points, [len(time_series)]])

    windows = [Window(start, end) for start, end in zip(diff_points[::2], diff_points[1::2])]
    return windows
