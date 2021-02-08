"""
===================
core.py
===================

Core classes and utilities that aren't specific to any part of the pipeline.

"""
import pathlib
from collections import namedtuple
from typing import NewType, Union

import numpy as np

__all__ = ["NumpyArrayLike", "Filepath", "Window"]

# Generic wrapper type for array-like data. Normally we'd use numpy's arraylike type, but that won't be available until
# Numpy 1.21: https://stackoverflow.com/questions/40378427/numpy-formal-definition-of-array-like-objects
NumpyArrayLike = NewType("NumpyArrayLike", np.ndarray)
# Generic path location, like a string or a pathlib.Path object.
Filepath = NewType("Filepath", Union[str, pathlib.os.PathLike])


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
