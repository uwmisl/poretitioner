"""
=========
filter.py
=========

# TODO : Write functionality for filtering nanopore captures : https://github.com/uwmisl/poretitioner/issues/43

"""
import re

import h5py
import numpy as np

from poretitioner import logger

from . import raw_signal_utils


def apply_feature_filters(signal, filters):
    """
    Check whether an array of current values (i.e. a single nanopore capture)
    passes a set of filters. Filters are based on summary statistics
    (e.g., mean) and a range of allowed values.

    Notes on filter behavior: If the filters dict is empty, there are no filters
    and the capture passes. Filters are inclusive of high and low values. Only
    supported filters are allowed. (mean, stdv, median, min, max, length)

    More complex filtering should be done with a custom function.

    TODO : Move filtering to its own module : (somewhat related: https://github.com/uwmisl/poretitioner/issues/43)

    Parameters
    ----------
    signal : array or list
        Time series of nanopore current values for a single capture.
    filters : dict
        Keys are strings matching the supported filters, values are a tuple
        giving the endpoints of the valid range. E.g. {"mean": (0.1, 0.5)}
        defines a filter such that 0.1 <= mean(capture) <= 0.5.

    Returns
    -------
    boolean
        True if capture passes all filters; False otherwise.
    """
    local_logger = logger.getLogger()
    # TODO: Implement logger best practices : https://github.com/uwmisl/poretitioner/issues/12
    supported_filters = {
        "mean": np.mean,
        "stdv": np.std,
        "median": np.median,
        "min": np.min,
        "max": np.max,
        "length": len,
    }
    other_filters = ["ejected"]
    pass_filters = True
    for filt, filt_vals in filters.items():
        if filt in supported_filters:
            low, high = filt_vals
            val = supported_filters[filt](signal)
            if (low is not None and low > val) or (high is not None and val > high):
                pass_filters = False
                return pass_filters
        elif filt in other_filters:
            continue
        else:
            local_logger.warning(f"Filter {filt} not supported; ignoring.")
    return pass_filters


def check_capture_ejection_by_read(f5, read_id):
    """Checks whether the current capture was in the pore until the voltage
    was reversed.

    Parameters
    ----------
    f5 : TODO
    read_id : TODO

    Returns
    -------
    boolean
        True if the end of the capture coincides with the end of a voltage window.
    """
    try:
        ejected = f5.get(f"/read_{read_id}/Signal").attrs["ejected"]
    except AttributeError:
        raise ValueError(f"path /read_{read_id} does not exist in the fast5 file.")
    return ejected


def check_capture_ejection(end_capture, voltage_ends, tol_obs=20):
    """Checks whether the current capture was in the pore until the voltage
    was reversed.

    Essentially checks whether a value (end_capture) is close enough (within
    a margin of tol_obs) to any value in voltage_ends.

    Parameters
    ----------
    end_capture : numeric
        The end time of the capture.
    voltage_ends : list of numeric
        List of times when the standard voltage ends.
    tol_obs : int, optional
        Tolerance for defining when the end of the capture = voltage end, by default 20

    Returns
    -------
    boolean
        True if the end of the capture coincides with the end of a voltage window.
    """
    for voltage_end in voltage_ends:
        if np.abs(end_capture - voltage_end) < tol_obs:
            return True
    return False


def apply_filters_to_read(config, f5, read_id, filter_name):
    passed_filters = True

    # Check whether the capture was ejected
    if "ejected" in config["filters"][filter_name]:
        only_use_ejected_captures = config["filters"][filter_name]["ejected"]  # TODO
        if only_use_ejected_captures:
            capture_ejected = check_capture_ejection_by_read(f5, read_id)
            if not capture_ejected:
                passed_filters = False
                return passed_filters
    else:
        only_use_ejected_captures = False  # could skip this, leaving to help read logic

    # Apply all the filters
    signal = raw_signal_utils.get_fractional_blockage_for_read(f5, read_id)
    print(config["filters"][filter_name])
    print(f"min = {np.min(signal)}")
    passed_filters = apply_feature_filters(signal, config["filters"][filter_name])
    return passed_filters


def filter_and_store_result(config, fast5_files, filter_name, overwrite=False):
    # Apply a new set of filters
    # Write filter results to fast5 file (using format)
    # if only_use_ejected_captures = config["???"]["???"], then check_capture_ejection
    # if there's a min length specified in the config, use that
    # if feature filters are specified, apply those
    # save all filter parameters in the filter_name path
    filter_path = f"/Filter/{filter_name}"

    # TODO: parallelize this (embarassingly parallel structure)
    for fast5_file in fast5_files:
        with h5py.File(fast5_file, "a") as f5:
            if overwrite is False and filter_path in f5:
                continue
            passed_read_ids = []
            for read_h5group_name in f5.get("/"):
                if "read" not in read_h5group_name:
                    continue
                read_id = re.findall(r"read_(.*)", read_h5group_name)[0]

                passed_filters = apply_filters_to_read(config, f5, read_id, filter_name)
                if passed_filters:
                    passed_read_ids.append(read_id)
            write_filter_results(f5, config, passed_read_ids, filter_name)


def write_filter_results(f5, config, read_ids, filter_name):
    local_logger = logger.getLogger()
    filter_path = f"/Filter/{filter_name}"
    if filter_path not in f5:
        f5.create_group(f"{filter_path}/pass")
    filt_grp = f5.get(filter_path)

    # Save filter configuration to the fast5 file at filter_path
    for k, v in config.items():
        if k == filter_name:
            local_logger.debug("keys and vals:", k, v)
            for filt, filt_vals in v.items():
                if len(filt_vals) == 2:
                    (min_filt, max_filt) = filt_vals
                    # Create compound dset for filters
                    local_logger.debug("filt types", type(min_filt), type(max_filt))
                    dtypes = np.dtype([("min", type(min_filt), ("max", type(max_filt)))])
                    d = filt_grp.create_dataset(k, (2,), dtype=dtypes)
                    d[filt] = (min_filt, max_filt)
                else:
                    d = filt_grp.create_dataset(k)
                    d[filt] = filt_vals

    # For all read_ids that passed the filter (AKA reads that were passed in),
    # create a hard link in the filter_path to the actual read's location in
    # the fast5 file.
    for read_id in read_ids:
        read_path = f"/read_{read_id}"
        read_grp = f5.get(read_path)
        local_logger.debug(read_grp)
        filter_read_path = f"{filter_path}/pass/read_{read_id}"
        # Create a hard link from the filter read path to the actual read path
        f5[filter_read_path] = read_grp


def filter_like_existing(config, example_fast5, example_filter_path, fast5_files, new_filter_path):
    # Filters a set of fast5 files exactly the same as an existing filter
    # TODO : #68 : implement
    raise NotImplementedError()
