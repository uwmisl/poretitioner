"""
===================
core.py
===================

Core classes and utilities that aren't specific to any part of the pipeline.

"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from os import PathLike
from typing import Dict, List, TypeVar, Union

import numpy as np

__all__ = [
    "find_windows_below_threshold",
    "NumpyArrayLike",
    "PathLikeOrString",
    "Window",
    "WindowsByChannel",
]

# Generic wrapper type for array-like data. Normally we'd use numpy's arraylike type, but that won't be available until
# Numpy 1.21: https://stackoverflow.com/questions/40378427/numpy-formal-definition-of-array-like-objects
NumpyArrayLike = np.ndarray

# Represent a path or a string representing a path.
PathLikeOrString = Union[str, PathLike]


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

    def overlaps(self, regions: List[Window]) -> List[Window]:
        """Finds all of the regions in the given list that overlap with the window.
        Needs to have at least one overlapping point; cannot be just adjacent.
        Incomplete overlaps are returned.

        Parameters
        ----------
        regions : List[Window] of windows.
            Start and end points to check for overlap with the specified window.
            All regions are assumed to be mutually exclusive and sorted in
            ascending order.

        Returns
        -------
        overlapping_regions : List[Window]
            Regions that overlap with the window in whole or part, returned in
            ascending order.

        Raises
        ------
        ValueError
            Somehow a region was neither before a window, after a window, or overlapping a window
            This can only happen if one of the region windows was invalid (e.g. has a start greater
            than end, violating locality).
        """

        window_start, window_end = self
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
                e = f"Shouldn't have gotten here! Region: {region}, Window: {self}."
                raise ValueError(e)
        return overlapping_regions


def find_windows_below_threshold(
    time_series: NumpyArrayLike, threshold: float | int
) -> List[Window]:
    """
    Find regions where the time series data points drop at or below the
    specified threshold.

    Parameters
    ----------
    time_series : NumpyArrayLike
        Array containing time series data points.
    threshold : float or int
        Find regions where the time series drops at or below this value

    Returns
    -------
    list of Windows (start, end): List[Window]
        Each item in the list represents the (start, end) points of regions
        where the input array drops at or below the threshold.
    """

    # Katie Q: I still don't understand how this function works haha. Let's talk next standup?
    diff_points = np.where(np.abs(np.diff(np.where(time_series <= threshold, 1, 0))) == 1)[0]
    if time_series[0] <= threshold:
        diff_points = np.hstack([[0], diff_points])
    if time_series[-1] <= threshold:
        diff_points = np.hstack([diff_points, [time_series.duration]])
    return [Window(start, end) for start, end in zip(diff_points[::2], diff_points[1::2])]


@dataclass
class WindowsByChannel:
    by_channel: Dict[int, List[Window]]

    def __init__(self, *args):
        self.by_channel = dict()

    def __getitem__(self, channel_number: int):
        return self.by_channel[channel_number]

    def __setitem__(self, channel_number: int, windows: List[Window]):
        self.by_channel[channel_number] = windows

    def keys(self):
        return self.by_channel.keys()
