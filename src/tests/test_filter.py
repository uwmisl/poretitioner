"""
================
test_filtering.py
================

This module contains tests for True  # filtering.py functionality.

"""
import os
import re
from shutil import copyfile

import h5py
import pytest

import src.poretitioner.utils.filtering as filtering


# TODO: Restore Filter unit tests: https://github.com/uwmisl/poretitioner/issues/88
def apply_feature_filters_empty_test():
    """Check for pass when no valid filters are provided."""
    # capture -- mean: 1; stdv: 0; median: 1; min: 1; max: 1; len: 6
    capture = [1, 1, 1, 1, 1, 1]
    filters = []
    # No filter given -- pass
    pass_filters = filtering.apply_feature_filters(capture, filters)
    assert pass_filters


def apply_feature_filters_length_test():
    """Test length filter function."""
    # capture -- mean: 1; stdv: 0; median: 1; min: 1; max: 1; len: 6
    capture = [1, 1, 1, 1, 1, 1]

    # Only length filter -- pass (edge case, inclusive high)
    filters = [filtering.LengthFilter(0, 6)]
    pass_filters = filtering.apply_feature_filters(capture, filters)
    assert pass_filters

    # Only length filter -- pass (edge case, inclusive low)
    filters = [filtering.LengthFilter(6, 10)]
    pass_filters = filtering.apply_feature_filters(capture, filters)
    assert pass_filters

    # Only length filter -- fail (too short)
    filters = [filtering.LengthFilter(8, 10)]
    pass_filters = filtering.apply_feature_filters(capture, filters)
    assert not pass_filters

    # Only length filter -- fail (too long)
    filters = [filtering.LengthFilter(0, 5)]
    pass_filters = filtering.apply_feature_filters(capture, filters)
    assert not pass_filters

    # Only length filter -- pass (no filter actually given)
    filters = [filtering.LengthFilter(None, None)]
    pass_filters = filtering.apply_feature_filters(capture, filters)
    assert pass_filters


def apply_feature_filters_mean_test():
    """Test mean filter function. stdv, median, min, and max apply similarly."""
    # capture -- mean: 0.5; stdv: 0.07; median: 0.5; min: 0.4; max: 0.6; len: 5
    capture = [0.5, 0.5, 0.6, 0.4, 0.5]
    # Only mean filter -- pass
    filters = [filtering.MeanFilter(0, 1)]
    pass_filters = filtering.apply_feature_filters(capture, filters)
    assert pass_filters

    # Only mean filter -- fail (too high)
    filters = [filtering.MeanFilter(0, 0.4)]
    pass_filters = filtering.apply_feature_filters(capture, filters)
    assert not pass_filters

    # Only mean filter -- fail (too low)
    filters = [filtering.MeanFilter(0.6, 1)]
    pass_filters = filtering.apply_feature_filters(capture, filters)
    assert not pass_filters


def check_capture_ejection_by_read_test():
    f5_fail = "src/tests/data/bulk_fast5_dummy.fast5"
    assert os.path.exists(f5_fail)
    bad_read_id = "akejwoeirjo;ewijr"
    with h5py.File(f5_fail, "r") as f5:
        with pytest.raises(ValueError) as e:
            filtering.check_capture_ejection_by_read(f5, bad_read_id)
            assert "does not exist in the fast5 file" in e
    # TODO implement fast5 writing to file


def check_capture_ejection_test():
    """Essentially checks whether a value (end_capture) is close enough (within
    a margin of tol_obs) to any value in voltage_ends.
    """
    end_capture = 1000
    voltage_ends = [0, 1000, 2000, 3000]
    tol_obs = 100
    assert filtering.check_capture_ejection(end_capture, voltage_ends, tol_obs=tol_obs)

    end_capture = 1200
    voltage_ends = [0, 1000, 2000, 3000]
    tol_obs = 100
    assert not filtering.check_capture_ejection(end_capture, voltage_ends, tol_obs=tol_obs)

    end_capture = 3100
    voltage_ends = [0, 1000, 2000, 3000]
    tol_obs = 100
    assert not filtering.check_capture_ejection(end_capture, voltage_ends, tol_obs=tol_obs)


# def apply_filters_to_read_test():
#     orig_f5_fname = "src/tests/data/reads_fast5_dummy_9captures.fast5"
#     filter_f5_fname = "apply_filters_to_read_test.fast5"
#     copyfile(orig_f5_fname, filter_f5_fname)

#     standard_filter = {
#         "mean": (0.05, 0.9),
#         "min": (0.001, 0.9),
#         "length": (None, 100_000),
#         "median": (0.05, 0.9),
#         "stdv": (0.01, 0.5),
#         "ejected": False,
#     }
#     config = {"filters": {"standard filter": standard_filter}}
#     pass_reads = [
#         "697de4c1-1aef-41b9-ae0d-d676e983cb7e",
#         "8e8181d2-d749-4735-9cab-37648b463f88",
#         "a0c40f5a-c685-43b9-a3b7-ca13aa90d832",
#         "cd6fa746-e93b-467f-a3fc-1c9af815f836",
#         "f5d76520-c92b-4a9c-b5cb-a04414db527e",
#     ]

#     filter_name = "standard filter"
#     with h5py.File(filter_f5_fname, "r") as f5:
#         for g in f5.get("/"):
#             if "read" in g:
#                 read_id = re.findall(r"read_(.*)", str(g))[0]
#                 # TODO: Pipe through filtering https://github.com/uwmisl/poretitioner/issues/43 https://github.com/uwmisl/poretitioner/issues/68
#                 passed = True  # filtering.apply_filters_to_read(config, f5, read_id, filter_name)
#                 if passed:
#                     assert read_id in pass_reads
#                 else:
#                     assert read_id not in pass_reads
#     os.remove(filter_f5_fname)


# def filter_and_store_result_test():
#     # TODO docstring
#     # Examine the 9 captures in the test file
#     # Create a set of filters that removes some and keeps others
#     # Call fn
#     # Verify which reads passed the filters

#     orig_f5_fname = "src/tests/data/reads_fast5_dummy_9captures.fast5"
#     filter_f5_fname = "filter_and_store_result_test.fast5"
#     copyfile(orig_f5_fname, filter_f5_fname)

#     standard_filter = {
#         "mean": (0.05, 0.9),
#         "min": (0.001, 0.9),
#         "length": (None, 100_000),
#         "median": (0.05, 0.9),
#         "stdv": (0.01, None),
#         "ejected": False,
#     }
#     lenient_filter = {"ejected": False}
#     strict_filter = {"mean": (1, 1), "min": (1, 1), "ejected": True}
#     config = {
#         "filters": {
#             "standard filter": standard_filter,
#             "lenient filter": lenient_filter,
#             "strict filter": strict_filter,
#         }
#     }

#     filter_names = ["standard filter", "lenient filter", "strict filter"]
#     n_passing_filters = [5, 9, 0]
#     for filter_name in filter_names:
#         # TODO: Pipe through filtering https://github.com/uwmisl/poretitioner/issues/43 https://github.com/uwmisl/poretitioner/issues/68
#         True  # filtering.filter_and_store_result(config, [filter_f5_fname], filter_name, overwrite=True)

#     with h5py.File(filter_f5_fname, "r") as f5:
#         for i, filter_name in enumerate(filter_names):
#             passing_reads = list(f5.get(f"/Filter/{filter_name}/pass"))
#             assert len(passing_reads) == n_passing_filters[i]
#     os.remove(filter_f5_fname)


# def write_filter_results_test():
#     # Copy a tester fast5 file
#     orig_f5_fname = "src/tests/data/reads_fast5_dummy_9captures.fast5"
#     filter_f5_fname = "write_filter_results_test.fast5"
#     copyfile(orig_f5_fname, filter_f5_fname)
#     # Define config dict that contains filter info
#     config = {
#         "compute": {"n_workers": 4},
#         "segment": {
#             "voltage_threshold": -180,
#             "signal_threshold": 0.7,
#             "translocation_delay": 10,
#             "open_channel_prior_mean": 230,
#             "open_channel_prior_stdv": 25,
#             "good_channels": [1, 2, 3],
#             "end_tol": 0,
#             "terminal_capture_only": False,
#         },
#         "filters": {
#             "base filter": {"length": (100, None)},
#             "test filter": {"min": (100, None)},
#         },
#         "output": {"capture_f5_dir": "src/tests/", "captures_per_f5": 1000},
#     }

#     # Get read ids from the original file, and take some that "passed" our
#     # imaginary True  # filtering.
#     read_ids = []
#     with h5py.File(filter_f5_fname, "r") as f5:
#         for g in f5.get("/"):
#             if "read" in str(g):
#                 read_id = re.findall(r"read_(.*)", str(g))[0]
#                 read_ids.append(read_id)
#     read_ids = read_ids[1 : len(read_ids) : 2]

#     # Call fn
#     filter_name = "test filter"
#     with h5py.File(filter_f5_fname, "a") as f5:
#         # TODO: Pipe through filtering https://github.com/uwmisl/poretitioner/issues/43 https://github.com/uwmisl/poretitioner/issues/68
#         True  # filtering.write_filter_results(f5, config, read_ids, filter_name)

#     # Check:
#     #   * len(read_ids) is correct
#     #   * config values are all present (TODO)
#     with h5py.File(filter_f5_fname, "r") as f5:
#         g = list(f5.get(f"/Filter/{filter_name}/pass"))
#         g = [x for x in g if "read" in x]
#         assert len(g) == len(read_ids)

#     os.remove(filter_f5_fname)
#     # TODO other versions:
#     #   * read id is not present in the file (what behavior?)
