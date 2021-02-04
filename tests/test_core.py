import pytest
from poretitioner.utils.core import Window


def window_duration_test():
    """Test that window duration is the difference between end and start times.
    """
    start = 5
    end = 11
    expected_duration = 6  # 11 - 5
    window = Window(start, end)
    assert window.duration == expected_duration


def window_invalid_start_end_test():
    """Test that windows with start times after end times are invalid.
    """
    start = 10
    end = 5  # Can't go back in time!
    with pytest.raises(ValueError):
        Window(start, end)
