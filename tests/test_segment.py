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
