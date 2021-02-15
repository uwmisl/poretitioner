"""
========================
test_raw_signal_utils.py
========================

This module contains tests for raw_signal_utils.py functionality.

"""
import h5py

import poretitioner.utils.raw_signal_utils as raw_signal_utils


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


def unscale_raw_current_test():
    """Test ability to convert back & forth between digital data & pA."""
    bulk_f5_fname = "tests/data/bulk_fast5_dummy.fast5"
    channel_no = 1
    with h5py.File(bulk_f5_fname, "r") as f5:
        offset, rng, digi = raw_signal_utils.get_scale_metadata(f5, channel_no)
        raw_orig = raw_signal_utils.get_raw_signal(f5, channel_no)
    raw_pA = raw_signal_utils.scale_raw_current(raw_orig, offset, rng, digi)
    raw_digi = raw_signal_utils.digitize_raw_current(raw_pA, offset, rng, digi)
    for x, y in zip(raw_orig, raw_digi):
        assert abs(x - y) <= 1  # allow for slight rounding errors


def get_overlapping_regions_test():
    window = (10, 100)
    excl_regions = [(0, 9), (9, 10), (100, 101), (1000, 1001)]
    overlap = raw_signal_utils.get_overlapping_regions(window, excl_regions)
    assert len(overlap) == 0
    incl_regions = [(9, 11), (20, 40), (99, 100), (99, 1000)]
    overlap = raw_signal_utils.get_overlapping_regions(window, incl_regions)
    assert len(overlap) == len(incl_regions)
