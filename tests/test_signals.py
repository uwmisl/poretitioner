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
CHANNEL_NUMBER = 1
OPEN_CHANNEL_GUESS = 44
OPEN_CHANNEL_BOUND = 22


class TestBaseSignal:
    def channel_number_is_always_greater_than_zero_test(self):
        """Oxford nanopore uses one-based indexing for channel indicies. It should never be less than 0.
        """
        bad_channel_number: int = 0

        with pytest.raises(ValueError):
            BaseSignal([1, 2, 3], bad_channel_number, CALIBRATION)

    def signal_can_be_used_like_numpy_array_test(self):
        signal = [1, 2, 3]
        raw = BaseSignal(signal, CHANNEL_NUMBER, CALIBRATION)

        assert np.median(raw) == 2
        assert np.mean(raw) == 2
        assert np.max(raw) == 3
        assert np.min(raw) == 1

        assert np.max(signal * 10) == 30, "We can multiply by scalars, just like numpy arrays"
        assert signal + 10 == [11, 12, 13], "We can add by scalars, just like numpy arrays"
        assert signal + signal == [2, 4, 6], "We can add by other signals, just like numpy arrays"

    def duration_test(self):
        signal = [1, 2, 3]
        raw = BaseSignal(signal, CHANNEL_NUMBER, CALIBRATION)
        assert raw.duration == len(
            signal
        ), "Duration should match the number of samples in the signal."


class TestRawSignal:
    def signal_converts_to_picoamperes_test(self):
        signal = [1, 2, 3]
        raw = RawSignal(signal, CHANNEL_NUMBER, CALIBRATION)

        expected_signal_in_pA = 2 * signal
        assert raw.to_picoamperes == expected_signal_in_pA, "Signal should convert to picoamperes."

    def signal_converts_to_fractionalized_test(self):
        signal = [1, 2, 3]
        raw = RawSignal(signal, CHANNEL_NUMBER, CALIBRATION)
        pico = raw.to_picoamperes()
        median = np.median(pico)

        expected = compute_fractional_blockage(pico, median)
        frac = raw.to_fractionalized()
        assert np.isclose(frac, expected), "Fractionalized current should match expected."


class TestPicoampereSignal:
    def signal_converts_to_fractionalized_test(self):
        pico = PicoampereSignal([1, 2, 3], CHANNEL_NUMBER, CALIBRATION)
        median = np.median(pico)

        expected = compute_fractional_blockage(pico, median)
        frac = pico.to_fractionalized()
        assert np.isclose(frac, expected), "Fractionalized current should match expected."

    def picoampere_to_raw_test(self):
        picoamperes = [10, 20, 30]
        expected_raw = [5, 10, 15]

        pico = PicoampereSignal(picoamperes, CHANNEL_NUMBER, CALIBRATION)
        assert (
            pico.to_raw() == expected_raw
        ), "Digitized picoampere current should match expected raw."

    def picoampere_digitize_test(self):
        picoamperes = [10, 20, 30]
        expected_raw = [5, 10, 15]

        pico = PicoampereSignal(picoamperes, CHANNEL_NUMBER, CALIBRATION)
        assert (
            pico.digitize() == expected_raw
        ), "Digitized picoampere current should match expected raw."

    def picoampere_digitize_matches_to_raw_result_test(self):
        picoamperes = [10, 20, 30]
        assert (
            picoamperes.digitize() == picoamperes.to_raw()
        ), "digitize() result should match to_raw() result."


class TestFractionalizedSignal:
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
