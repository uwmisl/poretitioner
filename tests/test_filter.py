"""
================
test_filter.py
================

This module contains tests for filter.py functionality.

"""
import os

import h5py
import numpy as np
import pytest

import poretitioner.utils.filter as filter


def apply_feature_filters_empty_test():
    """Check for pass when no valid filters are provided."""
    # capture -- mean: 1; stdv: 0; median: 1; min: 1; max: 1; len: 6
    capture = [1, 1, 1, 1, 1, 1]
    filters = {}
    # No filter given -- pass
    pass_filters = filter.apply_feature_filters(capture, filters)
    filters = {"not_a_filter": (0, 1)}
    # No *valid* filter given -- pass
    pass_filters = filter.apply_feature_filters(capture, filters)
    assert pass_filters


def apply_feature_filters_length_test():
    """Test length filter function."""
    # capture -- mean: 1; stdv: 0; median: 1; min: 1; max: 1; len: 6
    capture = [1, 1, 1, 1, 1, 1]

    # Only length filter -- pass (edge case, inclusive high)
    filters = {"length": (0, 6)}
    pass_filters = filter.apply_feature_filters(capture, filters)
    assert pass_filters

    # Only length filter -- pass (edge case, inclusive low)
    filters = {"length": (6, 10)}
    pass_filters = filter.apply_feature_filters(capture, filters)
    assert pass_filters

    # Only length filter -- fail (too short)
    filters = {"length": (8, 10)}
    pass_filters = filter.apply_feature_filters(capture, filters)
    assert not pass_filters

    # Only length filter -- fail (too long)
    filters = {"length": (0, 5)}
    pass_filters = filter.apply_feature_filters(capture, filters)
    assert not pass_filters

    # Only length filter -- pass (no filter actually given)
    filters = {"length": (None, None)}
    pass_filters = filter.apply_feature_filters(capture, filters)
    assert pass_filters


def apply_feature_filters_mean_test():
    """Test mean filter function. stdv, median, min, and max apply similarly."""
    # capture -- mean: 0.5; stdv: 0.07; median: 0.5; min: 0.4; max: 0.6; len: 5
    capture = [0.5, 0.5, 0.6, 0.4, 0.5]
    # Only mean filter -- pass
    filters = {"mean": (0, 1)}
    pass_filters = filter.apply_feature_filters(capture, filters)
    assert pass_filters

    # Only mean filter -- fail (too high)
    filters = {"mean": (0, 0.4)}
    pass_filters = filter.apply_feature_filters(capture, filters)
    assert not pass_filters

    # Only mean filter -- fail (too low)
    filters = {"mean": (0.6, 1)}
    pass_filters = filter.apply_feature_filters(capture, filters)
    assert not pass_filters


def check_capture_ejection_by_read_test():
    # TODO
    assert True is False


def check_capture_ejection_test():
    # TODO
    assert True is False


def filter_and_store_result_test():
    # TODO
    assert True is False


def write_filter_results_test():
    # TODO
    assert True is False
