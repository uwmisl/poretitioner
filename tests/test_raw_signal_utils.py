"""
========================
test_raw_signal_utils.py
========================

This module contains tests for raw_signal_utils.py functionality.

"""
import h5py
import poretitioner.utils.raw_signal_utils as raw_signal_utils
from poretitioner.signals import digitize_current


def get_voltage_test():
    bulk_f5_fname = "tests/data/bulk_fast5_dummy.fast5"
    # bulk_f5_fname = "tests/data/DESKTOP_CHF4GRO_20200110_FAL55785_MN25769_sequencing_run_01_10_20_run_01_44998.fast5"
    with h5py.File(bulk_f5_fname, "r") as f5:
        voltage = raw_signal_utils.get_voltage(f5)
    assert len(voltage == 500000)
    assert max(voltage) == 140
    assert min(voltage) == -180


def get_voltage_segment_test():
    bulk_f5_fname = "tests/data/bulk_fast5_dummy.fast5"
    start, end = 1000, 100000
    with h5py.File(bulk_f5_fname, "r") as f5:
        voltage = raw_signal_utils.get_voltage(f5, start=start, end=end)
    assert len(voltage) == end - start
    assert max(voltage) == 140
    assert min(voltage) == -180
