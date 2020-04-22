# This module is a placeholder demonstrating that our tests run.
import os
import h5py
import pytest
import poretitioner.utils.segment as segment


"""
    * Bulk fast5 doesn't have the required data sections
    * Config
        * throws error if None
        * throws error if doesn't have the right values
        * populated in fast5 if given
    * sub run
        * is None (no tuple)
        * adds extra info when given"""


def create_capture_fast5_test():
    """Test valid capture fast5 produced. Conditions:
    * Valid bulk fast5 input
    * Valid capture fast5 path
    * Empty segmenter config dict (valid)
    * No sub runs
    """
    bulk_f5_fname = "tests/data/bulk_fast5_dummy.fast5"
    capture_f5_fname = "tests/data/capture_fast5_dummy.fast5"
    config = {}
    segment.create_capture_fast5(bulk_f5_fname, capture_f5_fname, config,
                                 overwrite=True, sub_run=None)
    assert os.path.exists(capture_f5_fname)
    with h5py.File(capture_f5_fname, "r") as f5:
        assert f5["Meta/context_tags"] is not None
        assert "bulk_filename" in list(f5["Meta/context_tags"].attrs)
        assert f5["Meta/tracking_id"] is not None
        assert "sub_run_id" not in list(f5["Meta/tracking_id"].attrs)
        assert "sub_run_offset" not in list(f5["Meta/tracking_id"].attrs)
        assert "sub_run_duration" not in list(f5["Meta/tracking_id"].attrs)
    os.remove(capture_f5_fname)


def create_capture_fast5_overwrite_test():
    """Test overwriting capture fast5. Conditions:
    * Valid bulk fast5 input
    * Valid capture fast5 path (which exists)
    * Empty segmenter config dict (valid)
    * No sub runs
    * Test with and without overwrite flag
    """
    bulk_f5_fname = "tests/data/bulk_fast5_dummy.fast5"
    capture_f5_fname = "tests/data/capture_fast5_dummy_exists.fast5"
    with open(capture_f5_fname, "w") as f:
        f.write("\n")
    config = {}
    with pytest.raises(FileExistsError):
        segment.create_capture_fast5(bulk_f5_fname, capture_f5_fname, config,
                                     overwrite=False, sub_run=None)
    assert os.path.exists(capture_f5_fname)
    segment.create_capture_fast5(bulk_f5_fname, capture_f5_fname, config,
                                 overwrite=True, sub_run=None)
    assert os.path.exists(capture_f5_fname)
    os.remove(capture_f5_fname)


def create_capture_fast5_bulk_exists_test():
    """Test error thrown when bulk fast5 does not exist.
    """
    bulk_f5_fname = "tests/data/bulk_fast5_dummy_fake.fast5"
    capture_f5_fname = "tests/data/capture_fast5_dummy_bulkdemo.fast5"
    config = {}
    with pytest.raises(OSError):
        segment.create_capture_fast5(bulk_f5_fname, capture_f5_fname, config,
                                     overwrite=False, sub_run=None)


def create_capture_fast5_capture_path_missing_test():
    """Test error thrown when bulk fast5 does not exist.
    """
    bulk_f5_fname = "tests/data/bulk_fast5_dummy_fake.fast5"
    capture_f5_fname = "tests/DNE/capture_fast5_dummy_bulkdemo.fast5"
    config = {}
    with pytest.raises(OSError):
        segment.create_capture_fast5(bulk_f5_fname, capture_f5_fname, config,
                                     overwrite=False, sub_run=None)


def create_capture_fast5_subrun_test():
    """Test valid capture fast5 produced. Conditions:
    * Valid bulk fast5 input
    * Valid capture fast5 path
    * Empty segmenter config dict (valid)
    * HAS sub run
    """
    bulk_f5_fname = "tests/data/bulk_fast5_dummy.fast5"
    capture_f5_fname = "tests/data/capture_fast5_dummy_sub.fast5"
    config = {}
    segment.create_capture_fast5(bulk_f5_fname, capture_f5_fname, config,
                                 overwrite=True, sub_run=("S8w9g33", 9999999,
                                                          1000000))
    with h5py.File(capture_f5_fname, "r") as f5:
        assert "sub_run_id" in list(f5["Meta/tracking_id"].attrs)
        assert "S8w9g33" in str(f5["Meta/tracking_id"].attrs["sub_run_id"])
        assert "sub_run_offset" in list(f5["Meta/tracking_id"].attrs)
        assert f5["Meta/tracking_id"].attrs["sub_run_offset"] == 9999999
        assert "sub_run_duration" in list(f5["Meta/tracking_id"].attrs)
        assert f5["Meta/tracking_id"].attrs["sub_run_duration"] == 1000000
    os.remove(capture_f5_fname)


def apply_capture_filters_empty_test():
    """Check for pass when no valid filters are provided."""
    # capture -- mean: 1; stdv: 0; median: 1; min: 1; max: 1; len: 6
    capture = [1, 1, 1, 1, 1, 1]
    filters = {}
    # No filter given -- pass
    pass_filters = segment.apply_capture_filters(capture, filters)
    filters = {"not_a_filter": (0, 1)}
    # No *valid* filter given -- pass
    pass_filters = segment.apply_capture_filters(capture, filters)
    assert pass_filters


def apply_capture_filters_length_test():
    """Test length filter function."""
    # capture -- mean: 1; stdv: 0; median: 1; min: 1; max: 1; len: 6
    capture = [1, 1, 1, 1, 1, 1]

    # Only length filter -- pass (edge case, inclusive high)
    filters = {"length": (0, 6)}
    pass_filters = segment.apply_capture_filters(capture, filters)
    assert pass_filters

    # Only length filter -- pass (edge case, inclusive low)
    filters = {"length": (6, 10)}
    pass_filters = segment.apply_capture_filters(capture, filters)
    assert pass_filters

    # Only length filter -- fail (too short)
    filters = {"length": (8, 10)}
    pass_filters = segment.apply_capture_filters(capture, filters)
    assert not pass_filters

    # Only length filter -- fail (too long)
    filters = {"length": (0, 5)}
    pass_filters = segment.apply_capture_filters(capture, filters)
    assert not pass_filters


def apply_capture_filters_mean_test():
    """Test mean filter function. stdv, median, min, and max apply similarly."""
    # capture -- mean: 0.5; stdv: 0.07; median: 0.5; min: 0.4; max: 0.6; len: 5
    capture = [0.5, 0.5, 0.6, 0.4, 0.5]
    # Only mean filter -- pass
    filters = {"mean": (0, 1)}
    pass_filters = segment.apply_capture_filters(capture, filters)
    assert pass_filters

    # Only mean filter -- fail (too high)
    filters = {"mean": (0, 0.4)}
    pass_filters = segment.apply_capture_filters(capture, filters)
    assert not pass_filters

    # Only mean filter -- fail (too low)
    filters = {"mean": (0.6, 1)}
    pass_filters = segment.apply_capture_filters(capture, filters)
    assert not pass_filters


def find_captures_test():
    # TODO: Generate test data : https://github.com/uwmisl/poretitioner/issues/32
    # See issue for proposed test cases as well
    assert False
