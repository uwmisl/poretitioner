"""
=========
filter.py
=========

# TODO : Write functionality for filtering nanopore captures : https://github.com/uwmisl/poretitioner/issues/43

"""
import logging
import re

import h5py
import numpy as np

from . import raw_signal_utils

# def apply_length_filter(signal_len, length=20000):
#     return True if signal_len < length else False


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
    logger = logging.getLogger("apply_feature_filters")
    # TODO: Implement logger best practices : https://github.com/uwmisl/poretitioner/issues/12
    if logger.handlers:
        logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    supported_filters = {
        "mean": np.mean,
        "stdv": np.std,
        "median": np.median,
        "min": np.min,
        "max": np.max,
        "length": len,
    }
    pass_filters = True
    for filt, (low, high) in filters.items():
        if filt in supported_filters:
            val = supported_filters[filt](signal)
            if (low is not None and low > val) or (high is not None and val > high):
                pass_filters = False
                return pass_filters
        else:
            # Warn filter not supported
            logger.warning(f"Filter {filt} not supported; ignoring.")
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
    ejected = f5.get(f"/read_{read_id}").attrs["ejected"]
    return ejected


def check_capture_ejection(end_capture, voltage_ends, tol_obs=20):
    """Checks whether the current capture was in the pore until the voltage
    was reversed.

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


def filter_and_store_result(config, fast5_files, overwrite=False):
    # Apply a new set of filters
    # Write filter results to fast5 file (using format)
    # if only_use_ejected_captures = config["???"]["???"], then check_capture_ejection
    # if there's a min length specified in the config, use that
    # if feature filters are specified, apply those
    # save all filter parameters in the filter_name path
    filter_name = config["filter"]["filter_name"]
    filter_path = f"/Filter/{filter_name}"
    only_use_ejected_captures = config["???"]["???"]

    # TODO: parallelize this (embarassingly parallel structure)
    for fast5_file in fast5_files:
        with h5py.File(fast5_file, "w") as f5:
            if overwrite is False and filter_path in f5:
                continue
            passed_read_ids = []
            for read_h5group_name in f5.get("/"):
                read_id = re.findall(r"read_(.*)", read_h5group_name)[0]

                # Check whether the capture was ejected
                if only_use_ejected_captures:
                    capture_ejected = check_capture_ejection_by_read(f5, read_id)
                    if not capture_ejected:
                        continue

                # Apply all the filters
                passed_filters = True
                signal = raw_signal_utils.get_raw_signal_for_read(f5, read_id)
                passed_filters = apply_feature_filters(signal, config["filters"])
                if passed_filters:
                    passed_read_ids.append(read_id)
            write_filter_results(f5, config, passed_read_ids)


def write_filter_results(f5, config, read_ids):
    filter_name = config["filter"]["filter_name"]
    filter_path = f"/Filter/{filter_name}"
    if filter_path not in f5:
        f5.create_group(f"{filter_path}/pass")
    filt_grp = f5.get(filter_path)

    # Save filter configuration to the fast5 file at filter_path
    for k, v in config.items():
        if k == "filters":
            for filt, (min_filt, max_filt) in v.items():
                # Create compound dset for filters
                dtypes = np.dtype([("min", type(min_filt), ("max", type(max_filt)))])
                d = filt_grp.create_dataset(k, (2,), dtype=dtypes)
                d[filt] = (min_filt, max_filt)

    # For all read_ids that passed the filter (AKA reads that were passed in),
    # create a hard link in the filter_path to the actual read's location in
    # the fast5 file.
    for read_id in read_ids:
        read_path = f"/read_{read_id}"
        read_grp = f5.get(read_path)
        filter_read_path = f"{filter_path}/pass/read_{read_id}"
        # Create a hard link from the filter read path to the actual read path
        f5[filter_read_path] = read_grp


def filter_like_existing(config, example_fast5, example_filter_path, fast5_files, new_filter_path):
    # Filters a set of fast5 files exactly the same as an existing filter
    # TODO : #68 : implement
    pass
