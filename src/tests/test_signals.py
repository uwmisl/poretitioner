import pickle

import numpy as np
import pytest
from src.poretitioner.fast5s import BulkFile
from src.poretitioner.signals import (
    BaseSignal,
    Capture,
    Channel,
    ChannelCalibration,
    CurrentSignal,
    FractionalizedSignal,
    PicoampereSignal,
    RawSignal,
    VoltageSignal,
    compute_fractional_blockage,
    digitize_current,
)

# This calibration would take the raw current and just muliply it by 2.
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
    def signal_can_be_used_like_numpy_array_test(self):
        signal = np.array([1, 2, 3])
        base = BaseSignal(signal)

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
        n = 4
        adc_signal = np.array([2, 4, 8, 20])
        base = BaseSignal(adc_signal)
        assert not np.shares_memory(
            base, adc_signal, max_work=n
        ), "BaseSignal from an array should create new space in memory."

    def duration_test(self):
        signal = np.array([1, 2, 3])
        base = BaseSignal(signal)
        assert base.duration == len(
            base
        ), "Duration should match the number of samples in the signal."

    def can_serialize_test(self):
        signal = np.array([1, 2, 3])
        base = BaseSignal(signal)
        serialized = pickle.dumps(base)
        current_deserialized = pickle.loads(serialized)
        assert all(
            np.isclose(base, current_deserialized)
        ), "CurrentSignals can be serialized and deserialized."


class TestCurrentSignal:
    def channel_number_is_always_greater_than_zero_test(self):
        """Oxford nanopore uses one-based indexing for channel indicies. It should never be less than 0."""
        bad_channel_number: int = 0

        with pytest.raises(ValueError):
            CurrentSignal(np.array([1, 2, 3]), bad_channel_number, CALIBRATION)

    def can_serialize_test(self):
        signal = np.array([1, 2, 3])
        current = CurrentSignal(signal, CHANNEL_NUMBER, CALIBRATION)

        serialized = pickle.dumps(current)
        current_deserialized = pickle.loads(serialized)
        assert all(
            np.isclose(current, current_deserialized)
        ), "CurrentSignals can be serialized and deserialized."


class TestVoltageSignal:
    def can_serialize_test(self):
        signal = np.array([1, 2, 3])
        voltage = VoltageSignal(signal)

        serialized = pickle.dumps(voltage)
        voltage_deserialized = pickle.loads(serialized)
        assert all(
            np.isclose(voltage, voltage_deserialized)
        ), "VoltageSignals can be serialized and deserialized."

    def get_voltage_test(self):
        bulk_f5_fname = "src/tests/data/bulk_fast5_dummy.fast5"
        # bulk_f5_fname = "src/tests/data/DESKTOP_CHF4GRO_20200110_FAL55785_MN25769_sequencing_run_01_10_20_run_01_44998.fast5"
        with BulkFile(bulk_f5_fname, "r") as bulk:
            voltage = bulk.get_voltage()
            assert len(voltage) == 500_000
            assert max(voltage) == 140
            assert min(voltage) == -180

    def get_voltage_segment_test(self):
        bulk_f5_fname = "src/tests/data/bulk_fast5_dummy.fast5"
        start, end = 1000, 100_000
        with BulkFile(bulk_f5_fname, "r") as bulk:
            voltage = bulk.get_voltage(start=start, end=end)
            assert len(voltage) == end - start
            assert max(voltage) == 140
            assert min(voltage) == -180


class TestRawSignal:
    def signal_converts_to_picoamperes_test(self):
        signal = np.array([1, 2, 3])
        raw = RawSignal(signal, CHANNEL_NUMBER, CALIBRATION)

        expected_signal_in_pA = PicoampereSignal(
            signal * 2, CHANNEL_NUMBER, CALIBRATION
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

    def signal_converts_to_fractionalized_from_guess_test(self):
        signal = [1, 2, 3]
        channel_number = 2  # Arbitrary
        calibration = CALIBRATION
        raw = RawSignal(signal, channel_number, calibration)
        pico = raw.to_picoamperes()
        open_channel_pA = np.median(pico)

        expected = compute_fractional_blockage(pico, open_channel_pA)
        frac = raw.to_fractionalized_from_guess(
            open_channel_guess=2, open_channel_bound=4, default=2
        )
        assert all(
            np.isclose(frac, expected)
        ), "Fractionalized current should match expected."

    def converting_signal_to_fractionalized_from_guess_creates_new_array_test(self):
        # It's important that we do not share memory between raw/pico/fractional signals, so modifying one doesn't change the others unexpectedly.
        channel_number = 2  # Arbitrary
        n = 4
        adc_signal = np.array([2, 4, 8, 20])
        raw = RawSignal(adc_signal, channel_number, CALIBRATION)
        frac = raw.to_fractionalized_from_guess(
            open_channel_guess=8, open_channel_bound=5, default=2
        )
        assert not np.shares_memory(
            raw, frac, max_work=n
        ), "to_fractionalized_from_guess() should create a new space in memory."
        # Whew, we made it this far, so raw is treated as immutable.

    def can_convert_to_picoamperes_and_back_test(self):
        # Test that we can losslessly convert a raw signal to picoamperes, then back to raw.
        channel_number = 2  # Arbitrary
        adc_signal = np.array([2, 4, 8, 20])
        raw = RawSignal(adc_signal, channel_number, CALIBRATION)
        there_and_back = raw.to_picoamperes().to_raw()
        assert all(
            np.isclose(raw, there_and_back)
        ), "Should be able to convert raw to picoamperes and back to raw losslessly."

    def can_serialize_test(self):
        signal = np.array([1, 2, 3])
        raw = RawSignal(signal, CHANNEL_NUMBER, CALIBRATION)

        serialized = pickle.dumps(raw)
        base_deserialized = pickle.loads(serialized)
        assert all(
            np.isclose(raw, base_deserialized)
        ), "RawSignals can be serialized and deserialized."


class TestPicoampereSignal:
    def signal_converts_to_fractionalized_from_guess_test(self):
        pico = PicoampereSignal(PICO_SIGNAL, CHANNEL_NUMBER, CALIBRATION)
        median = np.median(pico)

        expected = compute_fractional_blockage(pico, median)
        frac = pico.to_fractionalized_from_guess(
            open_channel_guess=OPEN_CHANNEL_GUESS, open_channel_bound=OPEN_CHANNEL_BOUND
        )
        assert all(
            np.isclose(frac, expected)
        ), "Fractionalized current should match expected."


class PicoampereSignalTest:
    def __init__(self):
        self.calibration = CALIBRATION
        self.channel_number = 1

    def signal_converts_to_fractionalized_from_guess_test(self):
        pico = PicoampereSignal([1, 2, 3], self.channel_number, self.calibration)
        median = np.median(pico)

        expected = compute_fractional_blockage(pico, median)
        frac = pico.to_fractionalized_from_guess()
        assert all(
            np.isclose(frac, expected)
        ), "Fractionalized current should match expected."

    def picoampere_to_raw_test(self):
        picoamperes = [10, 20, 30]
        expected_raw = [5, 10, 15]

        pico = PicoampereSignal(picoamperes, self.channel_number, self.calibration)
        assert all(
            np.isclose(pico.to_raw(), expected_raw)
        ), "Digitized picoampere current should match expected raw."

    def picoampere_digitize_test(self):
        picoamperes = [10, 20, 30]
        expected_raw = [5, 10, 15]

        pico = PicoampereSignal(picoamperes, self.channel_number, self.calibration)
        assert all(
            np.isclose(pico.digitize(), expected_raw)
        ), "Digitized picoampere current should match expected raw."

    def picoampere_digitize_matches_to_raw_result_test(self):
        picoamperes = [10, 20, 30]
        pico = PicoampereSignal(picoamperes, self.channel_number, self.calibration)
        assert all(
            np.isclose(pico.digitize(), pico.to_raw())
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
    def unscale_raw_current_test(self):
        """Test ability to convert back & forth between digital data & pA."""
        channel_number = 1
        signal_pico = [220, 100, 4, 77, 90, 39, 1729, 369]
        pico = PicoampereSignal(signal_pico, channel_number, CALIBRATION)
        digitized_result = pico.digitize()
        raw_digi = digitize_current(signal_pico, CALIBRATION)
        assert np.isclose(digitized_result, raw_digi)
