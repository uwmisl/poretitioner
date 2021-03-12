import pytest
import src.poretitioner as poretitioner
from src.poretitioner.utils.core import Window


class WindowTest:
    def window_duration_test(self):
        """Test that window duration is the difference between end and start times.
        """
        start = 5
        end = 11
        expected_duration = 6  # 11 - 5
        window = Window(start, end)
        assert window.duration == expected_duration

    def window_invalid_start_end_test(self):
        """Test that windows with start times after end times have invalid durations.
        """
        start = 10
        end = 5  # Can't go back in time!
        with pytest.raises(ValueError):
            Window(start, end).duration

    def realistic_overlapping_regions_test(self):
        window = Window(10, 100)
        excl_regions = [(0, 9), (9, 10), (100, 101), (1000, 1001)]
        overlap = window.overlaps(excl_regions)
        assert (
            len(overlap) == 0
        ), "Window.overlaps should not return any windows when there's no overlap"
        incl_regions = [(9, 11), (20, 40), (99, 100), (99, 1000)]
        overlap = window.overlaps(incl_regions)
        assert len(overlap) == len(
            incl_regions
        ), "Window.overlaps should as many windows as overlap with the regions"
