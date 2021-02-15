import numpy as np
import pytest
from poretitioner.signals import (
    BaseSignal,
    Capture,
    Channel,
    ChannelCalibration,
    FractionalizedSignal,
    PicoampereSignal,
    RawSignal,
    compute_fractional_blockage,
)

# This calibration would take the raw current and just muliply it by 2.
CALIBRATION = ChannelCalibration(0, 2, 1)

OPEN_CHANNEL_GUESS = 44
OPEN_CHANNEL_BOUND = 22


class BaseSignalTest:
    def __init__(self):
        self.calibration = CALIBRATION

    def channel_number_is_always_greater_than_zero_test(self):
        """Oxford nanopore uses one-based indexing for channel indicies. It should never be less than 0.
        """
        channel_number: int = 0

        with pytest.raises(ValueError):
            BaseSignal([1, 2, 3], channel_number, self.calibration)

    def signal_can_be_used_like_numpy_array_test(self):
        signal = [1, 2, 3]
        raw = BaseSignal(signal)

        assert np.median(raw) == 2
        assert np.mean(raw) == 2
        assert np.max(raw) == 3
        assert np.min(raw) == 1

        assert np.max(signal * 10) == 30, "We can multiply by scalars, just like numpy arrays"
        assert signal + 10 == [11, 12, 13], "We can add by scalars, just like numpy arrays"
        assert signal + signal == [2, 4, 6], "We can add by other signals, just like numpy arrays"

    def duration_test(self):
        signal = [1, 2, 3]
        raw = BaseSignal(signal)
        assert raw.duration == len(
            signal
        ), "Duration should match the number of samples in the signal."


class RawSignalTest:
    def __init__(self):
        self.calibration = CALIBRATION
        self.channel_number = 1

    def signal_converts_to_picoamperes_test(self):
        signal = [1, 2, 3]
        raw = RawSignal(signal, self.channel_number, self.calibration)

        expected_signal_in_pA = 2 * signal
        assert raw.to_picoamperes == expected_signal_in_pA, "Signal should convert to picoamperes."

    def signal_converts_to_fractionalized_test(self):
        signal = [1, 2, 3]
        raw = RawSignal(signal, self.channel_number, self.calibration)
        pico = raw.to_picoamperes()
        median = np.median(pico)

        expected = compute_fractional_blockage(pico, median)
        frac = raw.to_fractionalized()
        assert np.isclose(frac, expected), "Fractionalized current should match expected."


class PicoampereSignalTest:
    def __init__(self):
        self.calibration = CALIBRATION
        self.channel_number = 1

    def signal_converts_to_fractionalized_test(self):
        pico = PicoampereSignal([1, 2, 3], self.channel_number, self.calibration)
        median = np.median(pico)

        expected = compute_fractional_blockage(pico, median)
        frac = pico.to_fractionalized()
        assert np.isclose(frac, expected), "Fractionalized current should match expected."

    def picoampere_to_raw_test(self):
        picoamperes = [10, 20, 30]
        expected_raw = [5, 10, 15]

        pico = PicoampereSignal(picoamperes, self.channel_number, self.calibration)
        assert (
            pico.to_raw() == expected_raw
        ), "Digitized picoampere current should match expected raw."

    def picoampere_digitize_test(self):
        picoamperes = [10, 20, 30]
        expected_raw = [5, 10, 15]

        pico = PicoampereSignal(picoamperes, self.channel_number, self.calibration)
        assert (
            pico.digitize() == expected_raw
        ), "Digitized picoampere current should match expected raw."

    def picoampere_digitize_matches_to_raw_result_test(self):
        picoamperes = [10, 20, 30]
        assert (
            picoamperes.digitize() == picoamperes.to_raw()
        ), "digitize() result should match to_raw() result."


class FractionalizedSignalTest:
    def signal_to_raw_test(self):
        assert True


class ChannelTest:
    def find_open_channel_current_yields_default_if_none_fall_into_bounds_test():
        assert True

    def find_open_channel_current_yields_median_test():
        assert True

    def find_open_channel_current_uses_10_percent_bound_by_default_test():
        assert True

    def find_open_channel_current_realistic_test():
        assert True


class HelpersTest:
    def unscale_raw_current_test():
        """Test ability to convert back & forth between digital data & pA."""
        bulk_f5_fname = "tests/data/bulk_fast5_dummy.fast5"
        channel_number = 1
        signal = [220, 100, 4, 77, 90, 39, 1729, 369]
        raw_orig = RawSignal(signal, channel_number, CALIBRATION)
        pico = PicoampereSignal(raw_orig)
        raw_digi = digitize_raw_current(pico, offset, rng, digi)
        for x, y in zip(raw_orig, raw_digi):
            assert abs(x - y) <= 1  # allow for slight rounding errors
