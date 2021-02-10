import pickle
from tempfile import TemporaryFile

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
    find_open_channel_current,
)

# This calibration would take the raw current and just muliply it by 2.
# Because calibration is calculated via (raw + calibration.offset) * (calibration.rng / calibration.digitisation)
CALIBRATION = ChannelCalibration(0, 2, 1)
CHANNEL_NUMBER = 1
OPEN_CHANNEL_GUESS = 45
OPEN_CHANNEL_BOUND = 10

"""
For bound == 10:
                           upper            guess   lower
                            (>=)              |      (>)
                             |                |       |
"""
PICO_SIGNAL = np.array([100, 55, 50, 48, 46, 45, 44, 35, 30, 0, 150])
# Open channel calculation should include 55, and exclude 35.
PICO_EXPECTED_MEDIAN_SLICE = np.median(PICO_SIGNAL[1:7])
# When using 10% of guess as the bounds, it includes all values between 40.5 and 49.5.
OPEN_CHANNEL_GUESS_10_PERCENT_OF_GUESS_MEDIAN_SLICE = np.median(PICO_SIGNAL[3:7])


class TestBaseSignal:
    def channel_number_is_always_greater_than_zero_test(self):
        """Oxford nanopore uses one-based indexing for channel indicies. It should never be less than 0.
        """
        bad_channel_number: int = 0

        with pytest.raises(ValueError):
            BaseSignal(np.array([1, 2, 3]), bad_channel_number, CALIBRATION)

    def signal_can_be_used_like_numpy_array_test(self):
        signal = np.array([1, 2, 3])
        base = BaseSignal(signal, CHANNEL_NUMBER, CALIBRATION)

        assert np.median(base) == 2
        assert np.mean(base) == 2
        assert np.max(base) == 3
        assert np.min(base) == 1

        assert (
            np.max(np.multiply(base, 10)) == 30
        ), "We can multiply by scalars, just like numpy arrays."
        assert all(
            np.isclose(base + 10, [11, 12, 13])
        ), "We can add by scalars, just like numpy arrays."
        assert all(
            np.isclose(base + base, [2, 4, 6])
        ), "We can add by other signals, just like numpy arrays."

    def converting_signal_to_picoamperes_creates_new_array_test(self):
        # It's important that we do not share memory between base/pico/fractional signals,
        channel_number = 2  # Arbitrary
        n = 4
        adc_signal = np.array([2, 4, 8, 20])
        base = BaseSignal(adc_signal, channel_number, CALIBRATION)
        assert not np.shares_memory(
            base, adc_signal, max_work=n
        ), "BaseSignal from an array should create new space in memory."

    def duration_test(self):
        signal = np.array([1, 2, 3])
        base = BaseSignal(signal, CHANNEL_NUMBER, CALIBRATION)
        assert base.duration == len(
            base
        ), "Duration should match the number of samples in the signal."

    def can_serialize_test(self):
        signal = np.array([1, 2, 3])
        base = BaseSignal(signal, CHANNEL_NUMBER, CALIBRATION)

        serialized = pickle.dumps(base)
        base_deserialized = pickle.loads(serialized)
        assert all(base == base_deserialized), "BaseSignals can be serialized and deserialized."


class TestRawSignal:
    def signal_converts_to_picoamperes_test(self):
        signal = np.array([1, 2, 3])
        raw = RawSignal(signal, CHANNEL_NUMBER, CALIBRATION)

        expected_signal_in_pA = PicoampereSignal(
            np.multiply(signal, 2), CHANNEL_NUMBER, CALIBRATION
        )
        resulting_signal_in_pA = raw.to_picoamperes()
        assert all(
            np.isclose(resulting_signal_in_pA, expected_signal_in_pA)
        ), "Signal should convert to picoamperes."

    def converting_signal_to_picoamperes_creates_new_array_test(self):
        # It's important that we do not share memory between raw/pico/fractional signals, so modifying one doesn't change the others unexpectedly.
        channel_number = 2  # Arbitrary
        n = 4
        adc_signal = np.array([2, 4, 8, 20])
        raw = RawSignal(adc_signal, channel_number, CALIBRATION)
        pico = raw.to_picoamperes()
        assert not np.shares_memory(
            raw, pico, max_work=n
        ), "RawSignal should create a new space in memory."
        # Whew, we made it this far, so raw is treated as immutable.

    def signal_converts_to_fractionalized_test(self):
        signal = np.array([1, 2, 3])
        raw = RawSignal(signal, CHANNEL_NUMBER, CALIBRATION)
        pico = raw.to_picoamperes()
        open_channel_pA = np.median(pico)

        expected = compute_fractional_blockage(pico, open_channel_pA)
        frac = raw.to_fractionalized(open_channel_guess=2, open_channel_bound=4, default=2)
        assert all(np.isclose(frac, expected)), "Fractionalized current should match expected."

    def converting_signal_to_fractionalized_creates_new_array_test(self):
        # It's important that we do not share memory between raw/pico/fractional signals, so modifying one doesn't change the others unexpectedly.
        channel_number = 2  # Arbitrary
        n = 4
        adc_signal = np.array([2, 4, 8, 20])
        raw = RawSignal(adc_signal, channel_number, CALIBRATION)
        frac = raw.to_fractionalized(open_channel_guess=8, open_channel_bound=5, default=2)
        assert not np.shares_memory(
            raw, frac, max_work=n
        ), "to_fractionalized() should create a new space in memory."
        # Whew, we made it this far, so raw is treated as immutable.

    def can_serialize_test(self):
        signal = np.array([1, 2, 3])
        raw = RawSignal(signal, CHANNEL_NUMBER, CALIBRATION)

        serialized = pickle.dumps(raw)
        base_deserialized = pickle.loads(serialized)
        assert all(raw == base_deserialized), "RawSignals can be serialized and deserialized."


class TestPicoampereSignal:
    def signal_converts_to_fractionalized_test(self):
        pico = PicoampereSignal(PICO_SIGNAL, CHANNEL_NUMBER, CALIBRATION)

        open_channel_pA = PICO_EXPECTED_MEDIAN_SLICE

        expected = compute_fractional_blockage(pico, open_channel_pA)

        frac = pico.to_fractionalized(
            open_channel_guess=OPEN_CHANNEL_GUESS, open_channel_bound=OPEN_CHANNEL_BOUND, default=2
        )
        assert all(np.isclose(frac, expected)), "Fractionalized current should match expected."

    def picoampere_to_raw_test(self):
        picoamperes = [10, 20, 30]
        expected_raw = [5, 10, 15]

        pico = PicoampereSignal(picoamperes, CHANNEL_NUMBER, CALIBRATION)
        assert all(
            np.isclose(pico.to_raw(), expected_raw)
        ), "Digitized picoampere current should match expected raw."

    def picoampere_digitize_test(self):
        picoamperes = np.array([10, 20, 30])
        expected_raw = RawSignal(np.array([5, 10, 15]), CHANNEL_NUMBER, CALIBRATION)

        pico = PicoampereSignal(picoamperes, CHANNEL_NUMBER, CALIBRATION)
        assert all(
            np.isclose(pico.digitize(), expected_raw)
        ), "Digitized picoampere current should match expected raw."

    def picoampere_digitize_matches_to_raw_result_test(self):
        picoamperes = np.array([10, 20, 30])
        picoamperes = PicoampereSignal(picoamperes, CHANNEL_NUMBER, CALIBRATION)
        assert all(
            np.isclose(picoamperes.digitize(), picoamperes.to_raw())
        ), "digitize() result should match to_raw() result."

    def picoamperes_to_raw_creates_new_memory_test(self):
        picoamperes = np.array([10, 20, 30])
        picoamperes = PicoampereSignal(picoamperes, CHANNEL_NUMBER, CALIBRATION)
        digitized = picoamperes.to_raw()
        assert not np.shares_memory(
            digitized, picoamperes, max_work=len(picoamperes)
        ), "'.to_raw()' should create new memory."

    def picoamperes_digitize_creates_new_memory_test(self):
        picoamperes = np.array([10, 20, 30])
        picoamperes = PicoampereSignal(picoamperes, CHANNEL_NUMBER, CALIBRATION)
        digitized = picoamperes.digitize()
        assert not np.shares_memory(
            digitized, picoamperes, max_work=len(picoamperes)
        ), "'.digitize()' should create new memory."

    def can_serialize_test(self):
        signal = np.array([1, 2, 3])
        pico = PicoampereSignal(signal, CHANNEL_NUMBER, CALIBRATION)

        serialized = pickle.dumps(pico)
        base_deserialized = pickle.loads(serialized)
        assert all(pico == base_deserialized), "RawSignals can be serialized and deserialized."


class TestFractionalizedSignal:
    def signal_to_picoamperes_test(self):
        raw_signal = np.array([1, 2, 3])
        raw = RawSignal(raw_signal, CHANNEL_NUMBER, CALIBRATION)
        pico = PicoampereSignal(raw.to_picoamperes(), CHANNEL_NUMBER, CALIBRATION)
        open_channel_pA = np.median(pico)

        frac = FractionalizedSignal(
            compute_fractional_blockage(pico, open_channel_pA),
            CHANNEL_NUMBER,
            CALIBRATION,
            open_channel_pA,
        )
        converted_frac = pico.to_fractionalized(
            open_channel_guess=2, open_channel_bound=max(pico), default=2
        )

        assert all(
            np.isclose(frac, converted_frac)
        ), "Initializing a FractionalizedSignal and pico.to_fractionalized() should return the same result for the same open channel current."

    def converting_signal_to_picoamperes_creates_new_array_test(self):
        # It's important that we do not share memory between raw/pico/fractional signals, so modifying one doesn't change the others unexpectedly.
        channel_number = 2  # Arbitrary
        n = 4
        signal = PICO_SIGNAL
        open_channel_pA = PICO_EXPECTED_MEDIAN_SLICE
        frac = FractionalizedSignal(
            compute_fractional_blockage(signal, open_channel_pA),
            channel_number,
            CALIBRATION,
            open_channel_pA,
        )
        pico = frac.to_picoamperes()
        assert not np.shares_memory(
            pico, frac, max_work=n
        ), "to_picoamperes() should create a new space in memory."
        # Whew, we made it this far, so raw is treated as immutable.

    def can_serialize_test(self):
        open_channel_pA = 1
        signal = np.array([0.1, 0.2, 0.3])
        frac = FractionalizedSignal(signal, CHANNEL_NUMBER, CALIBRATION, open_channel_pA)

        serialized = pickle.dumps(frac)
        frac_deserialized = pickle.loads(serialized)
        assert all(frac == frac_deserialized), "RawSignals can be serialized and deserialized."


class TestChannel:
    def find_open_channel_current_yields_default_if_none_fall_into_bounds_test(self):
        pico = PicoampereSignal(PICO_SIGNAL, CHANNEL_NUMBER, CALIBRATION)
        default = 300
        expected = default
        bound = 1
        guess_beyond_bounds = np.max(pico) + (2 * bound)
        result = find_open_channel_current(
            pico, open_channel_guess=guess_beyond_bounds, open_channel_bound=bound, default=default
        )
        assert (
            result == expected
        ), "find_open_channel_current() should use default when no current samples fit within bounds."

    def find_open_channel_current_yields_median_test(self):
        pico = PicoampereSignal(PICO_SIGNAL, CHANNEL_NUMBER, CALIBRATION)

        expected = PICO_EXPECTED_MEDIAN_SLICE
        result = find_open_channel_current(
            pico, open_channel_guess=OPEN_CHANNEL_GUESS, open_channel_bound=OPEN_CHANNEL_BOUND
        )
        assert (
            result == expected
        ), "find_open_channel_current() should return median of samples within bounds."

    def find_open_channel_current_uses_10_percent_bound_by_default_test(self):
        pico = PicoampereSignal(PICO_SIGNAL, CHANNEL_NUMBER, CALIBRATION)

        expected = OPEN_CHANNEL_GUESS_10_PERCENT_OF_GUESS_MEDIAN_SLICE
        OPEN_CHANNEL_GUESS = 45
        result = find_open_channel_current(pico, open_channel_guess=OPEN_CHANNEL_GUESS)
        assert (
            result == expected
        ), "find_open_channel_current() should use 10% of the guess when a bound isn't provided."

    def find_open_channel_current_realistic_test(self):
        pico = PicoampereSignal(PICO_SIGNAL, CHANNEL_NUMBER, CALIBRATION)

        open_channel_pA = PICO_EXPECTED_MEDIAN_SLICE
        expected = find_open_channel_current(
            pico, open_channel_guess=OPEN_CHANNEL_GUESS, open_channel_bound=OPEN_CHANNEL_BOUND
        )

        assert (
            expected == open_channel_pA
        ), f"find_open_channel_current({pico}, open_channel_guess={OPEN_CHANNEL_GUESS}, open_channel_bound={OPEN_CHANNEL_BOUND}) should be {open_channel_pA}"

    def compute_fractional_blockage_creates_new_memory_test(self):
        result = compute_fractional_blockage(PICO_SIGNAL, 1)
        assert not np.shares_memory(
            result, PICO_SIGNAL
        ), "compute_fractional_blockage() should create new space in memory."
