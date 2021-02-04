"""
===================
core.py
===================

Core classes and utilities that aren't specific to any part of the pipeline.

"""
from collections import namedtuple
from typing import NewType

import numpy as np

__all__ = ["NumpyArrayLike", "Channel", "ChannelCalibration", "Window"]


DEFAULT_OPEN_CHANNEL_GUESS = 220
DEFAULT_OPEN_CHANNEL_BOUND = 15

# Generic wrapper type for array-like data. Normally we'd use numpy's arraylike type, but that won't be available until
# Numpy 1.21: https://stackoverflow.com/questions/40378427/numpy-formal-definition-of-array-like-objects
NumpyArrayLike = NewType("NumpyArrayLike", np.ndarray)


class Window(namedtuple("Window", ["start", "end"])):
    """Represents a general window of time.

    Parameters
    ----------
    start : float
        When this window starts.

    end : float
        When this window ends.
        End should always be greater than start.
    """

    @property
    def duration(self):
        """How long this window represents, measured by the difference between the start and end times.

        Returns
        -------
        float
            Duration of a window.

        Raises
        ------
        ValueError
            If window is invalid due to end time being smaller than start time.
        """
        if self.start > self.end:
            raise ValueError(
                f"Invalid window: end {self.end} is less than start {self.start}. Start should always be less than end."
            )

        duration = self.end - self.start
        return duration


class ChannelCalibration(namedtuple("ChannelCalibration", ["offset", "range", "digitisation"])):
    """On the Oxford Nanopore devices, there's an analog to digital converter that converts the raw, unitless analog signal
    transforms the raw current. To convert these raw signals to a measurement of current in picoAmperes,
    we need to know certain values for the channel configuration.


    Physics      |            Device                |        Computer
    ---------------------------------------------------------------------------
    Raw signal  -->   Analog-to-Digitial-Converter -->   Signal in PicoAmperes (pA)
    (unitless)           +  Channel Configuration

    Parameters
    ----------
    offset : float
        Adjustment to the ADC to each 0pA.
        Essentially, this is the average value the device ADC gives when there's zero current.
        We subtract this from the raw ADC value to account for this, like zero'ing out a scale before weighing something.

    range: int
        The range of digitized picoAmperes that can be produced after the analog-to-digital conversion.

    digitisation: int
        This is the number of values that can even be produced the Analog-to-Digital. This exists because only a finite number of values can be represnted by a digitized signal.
    """


class Channel:
    calibration: ChannelCalibration
    open_channel_guess: int
    open_channel_bound: int

    def __init__(
        self,
        calibration,
        open_channel_guess=DEFAULT_OPEN_CHANNEL_GUESS,
        open_channel_bound=DEFAULT_OPEN_CHANNEL_BOUND,
    ):
        """Calibration data for the nanopore channel. This is used to convert AAnalog-to-Digital-Converter (ADC) signals
        to picoamperes.

        Parameters
        ----------
        calibration : ChannelCalibration
            Device Analog-to-Digital-Converter (ADC) calibration data. This is used for translating unitless raw signals to picoamperes.
        open_channel_guess : int, optional
            Approximate estimate of the open channel current value, by default 220
        open_channel_bound : int, optional
            Approximate estimate of the variance in open channel current value from
            channel to channel (AKA the range to search), by default 15
        """
        self.calibration = calibration
        self.open_channel_guess = open_channel_guess
        self.open_channel_bound = open_channel_bound

    def __repr__(self):
        formatted = f"Calibration: {repr(self.calibration)}: open channel guess: {self.open_channel_guess} open channel bound: {self.open_channel_bound}"
        return formatted
