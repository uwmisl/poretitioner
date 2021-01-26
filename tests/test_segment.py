"""
================
test_segment.py
================

This module contains tests for segment.py functionality.

"""
import os

import h5py
import numpy as np
import pytest

import poretitioner.utils.segment as segment


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
    segment.create_capture_fast5(
        bulk_f5_fname, capture_f5_fname, config, overwrite=True, sub_run=None
    )
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
        segment.create_capture_fast5(
            bulk_f5_fname, capture_f5_fname, config, overwrite=False, sub_run=None
        )
    assert os.path.exists(capture_f5_fname)
    segment.create_capture_fast5(
        bulk_f5_fname, capture_f5_fname, config, overwrite=True, sub_run=None
    )
    assert os.path.exists(capture_f5_fname)
    os.remove(capture_f5_fname)


def create_capture_fast5_bulk_exists_test():
    """Test error thrown when bulk fast5 does not exist."""
    bulk_f5_fname = "tests/data/bulk_fast5_dummy_fake.fast5"
    capture_f5_fname = "tests/data/capture_fast5_dummy_bulkdemo.fast5"
    config = {}
    with pytest.raises(OSError):
        segment.create_capture_fast5(
            bulk_f5_fname, capture_f5_fname, config, overwrite=False, sub_run=None
        )


def create_capture_fast5_capture_path_missing_test():
    """Test error thrown when bulk fast5 does not exist."""
    bulk_f5_fname = "tests/data/bulk_fast5_dummy_fake.fast5"
    capture_f5_fname = "tests/DNE/capture_fast5_dummy_bulkdemo.fast5"
    config = {}
    with pytest.raises(OSError):
        segment.create_capture_fast5(
            bulk_f5_fname, capture_f5_fname, config, overwrite=False, sub_run=None
        )


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
    segment.create_capture_fast5(
        bulk_f5_fname,
        capture_f5_fname,
        config,
        overwrite=True,
        sub_run=("S8w9g33", 9_999_999, 1_000_000),
    )
    with h5py.File(capture_f5_fname, "r") as f5:
        assert "sub_run_id" in list(f5["Meta/tracking_id"].attrs)
        assert "S8w9g33" in str(f5["Meta/tracking_id"].attrs["sub_run_id"])
        assert "sub_run_offset" in list(f5["Meta/tracking_id"].attrs)
        assert f5["Meta/tracking_id"].attrs["sub_run_offset"] == 9_999_999
        assert "sub_run_duration" in list(f5["Meta/tracking_id"].attrs)
        assert f5["Meta/tracking_id"].attrs["sub_run_duration"] == 1_000_000
    os.remove(capture_f5_fname)


def find_captures_0_single_capture_test():
    data_file = "tests/data/capture_windows/test_data_capture_window_0.txt.gz"
    data = np.loadtxt(data_file, delimiter="\t", comments="#")
    actual_captures = [(33822, 92691, True)]
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = {"length": (100, None)}
    delay = 0
    end_tol = 0
    captures, open_channel_pA = segment.find_captures(
        data,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == len(actual_captures)
    for test_capture in captures:
        assert test_capture in actual_captures


def find_captures_0_single_capture_terminal_test():
    data_file = "tests/data/capture_windows/test_data_capture_window_0.txt.gz"
    data = np.loadtxt(data_file, delimiter="\t", comments="#")
    actual_captures = [(33822, 92691, True)]
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = True
    filters = {"length": (100, None)}
    delay = 0
    end_tol = 0
    captures, open_channel_pA = segment.find_captures(
        data,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == len(actual_captures)
    for test_capture in captures:
        assert test_capture in actual_captures


def find_captures_1_double_capture_noterminal_test():
    """Example capture window contains 2 long captures, neither terminal.
    Also contains a few short blips.

    Test: terminal_capture_only = True returns no captures"""
    data_file = "tests/data/capture_windows/test_data_capture_window_1.txt.gz"
    data = np.loadtxt(data_file, delimiter="\t", comments="#")
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = True
    filters = {"length": (100, None)}
    delay = 0
    end_tol = 0
    captures, open_channel_pA = segment.find_captures(
        data,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 0


def find_captures_1_double_capture_noterminal_2_test():
    """Example capture window contains 2 long captures, neither terminal.
    Also contains a few short blips.

    Test: terminal_capture_only = False returns 2 captures"""
    data_file = "tests/data/capture_windows/test_data_capture_window_1.txt.gz"
    data = np.loadtxt(data_file, delimiter="\t", comments="#")
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = {"length": (100, None)}
    delay = 0
    end_tol = 0
    captures, open_channel_pA = segment.find_captures(
        data,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 2


def find_captures_2_nocaptures_test():
    """Example capture window contains no captures.

    Test: find_captures returns no captures"""
    data_file = "tests/data/capture_windows/test_data_capture_window_2.txt.gz"
    data = np.loadtxt(data_file, delimiter="\t", comments="#")
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = {"length": (100, None)}
    delay = 10
    end_tol = 0
    captures, open_channel_pA = segment.find_captures(
        data,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 0


def find_captures_3_multicapture_terminal_test():
    """Example capture window contains 1 long terminal capture & 2 medium/short captures.

    Tests: find_captures returns...
    1 capture when terminal_capture_only = True
    """
    data_file = "tests/data/capture_windows/test_data_capture_window_3.txt.gz"
    data = np.loadtxt(data_file, delimiter="\t", comments="#")
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = True
    filters = {"length": (100, None)}
    delay = 0
    end_tol = 0
    captures, open_channel_pA = segment.find_captures(
        data,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 1


def find_captures_3_multicapture_nonterminal_test():
    """Example capture window contains 1 long terminal capture & 1 medium capture.

    Tests: find_captures returns...
    3 captures when terminal_capture_only = False
    """
    data_file = "tests/data/capture_windows/test_data_capture_window_3.txt.gz"
    data = np.loadtxt(data_file, delimiter="\t", comments="#")
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = {"length": (100, None)}
    delay = 0
    end_tol = 0
    captures, open_channel_pA = segment.find_captures(
        data,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 2


def find_captures_4_unfolded_terminal_test():
    """Example capture window contains 1 long terminal capturethat has unfolded.

    Tests: find_captures returns 1 capture
    """
    data_file = "tests/data/capture_windows/test_data_capture_window_4.txt.gz"
    data = np.loadtxt(data_file, delimiter="\t", comments="#")
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = True
    filters = {"length": (100, None)}
    delay = 0
    end_tol = 0
    captures, open_channel_pA = segment.find_captures(
        data,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 1


def find_captures_5_unfolded_terminal_test():
    """Example capture window contains 1 long terminal capture. It was captured
    almost immediately, causing a very short open pore region.

    Tests: find_captures returns 1 capture
    """
    data_file = "tests/data/capture_windows/test_data_capture_window_5.txt.gz"
    data = np.loadtxt(data_file, delimiter="\t", comments="#")
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = True
    filters = {"length": (100, None)}
    delay = 0
    end_tol = 0
    captures, open_channel_pA = segment.find_captures(
        data,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 1
    # Rough check; should be ~229.05 & anything close is okay.
    # The function is nondeterministic & should return this exact value, but if
    # future changes are made, some tolerance can be allowed.
    assert open_channel_pA > 228.5 and open_channel_pA < 230


def find_captures_6_clog_no_open_channel_test():
    """Example capture window contains 1 long terminal capture. Open pore region
    is extremely, extremely short. Test by cutting off the open pore region.

    Tests: find_captures returns 1 capture; open pore returns alt value.
    """
    data_file = "tests/data/capture_windows/test_data_capture_window_6.txt.gz"
    data = np.loadtxt(data_file, delimiter="\t", comments="#")[100:]
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = {"length": (100, None)}
    delay = 0
    end_tol = 0
    captures, open_channel_pA = segment.find_captures(
        data,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 1
    # Rough check; should be ~229.05 & anything close is okay.
    # The function is nondeterministic & should return this exact value, but if
    # future changes are made, some tolerance can be allowed.
    assert open_channel_pA == 230


def find_captures_7_capture_no_open_channel_test():
    """Example capture window contains 1 long terminal capture. Open pore region
    is extremely, extremely short. Test by cutting off the open pore region.

    Tests: find_captures returns 1 capture; open pore returns alt value.
    """
    data_file = "tests/data/capture_windows/test_data_capture_window_7.txt.gz"
    data = np.loadtxt(data_file, delimiter="\t", comments="#")[100:]
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = {"length": (100, None)}
    delay = 0
    end_tol = 0
    captures, open_channel_pA = segment.find_captures(
        data,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 1
    # Rough check; should be ~229.05 & anything close is okay.
    # The function is nondeterministic & should return this exact value, but if
    # future changes are made, some tolerance can be allowed.
    assert open_channel_pA == 230


def find_captures_8_capture_no_open_channel_test():
    """Example capture window contains 2 captures: both long, 1 terminal.
    Test non-terminal long capture.

    Tests: find_captures returns 2 captures.
           Checks exact capture boundaries with delay = 3
    """
    data_file = "tests/data/capture_windows/test_data_capture_window_8.txt.gz"
    data = np.loadtxt(data_file, delimiter="\t", comments="#")
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = {"length": (100, None)}
    delay = 3
    end_tol = 0
    captures, open_channel_pA = segment.find_captures(
        data,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 2
    actual_captures = [(11310, 22098, False), (26617, 94048, True)]
    for test_capture in captures:
        assert test_capture in actual_captures


def parallel_find_captures_test():
    bulk_f5_fname = "tests/data/bulk_fast5_dummy.fast5"
    config = {
        "compute": {"n_workers": 4},
        "segment": {
            "voltage_threshold": -180,
            "signal_threshold": 0.7,
            "translocation_delay": 10,
            "open_channel_prior_mean": 230,
            "open_channel_prior_stdv": 25,
            "good_channels": [1, 2, 3],
            "end_tol": 0,
            "terminal_capture_only": False,
        },
        "filters": {"base filter": {"length": (100, None)}},
        "output": {"capture_f5_dir": "tests/", "captures_per_f5": 1000},
    }
    segment.parallel_find_captures(bulk_f5_fname, config)
    run_id = "d0befb838f5a9a966e3c559dc3a75a6612745849"
    actual_n_captures = 9
    n_captures = 0
    capture_f5_fname = f"tests/{run_id}_1.fast5"
    with h5py.File(capture_f5_fname, "r") as f5:
        for grp in f5.get("/"):
            if "read" not in grp:
                continue
            n_captures += 1
            d = f5[grp]
            a = d["Signal"].attrs
            start_time_local = a.get("start_time_local")
            start_time_bulk = a.get("start_time_bulk")
            assert start_time_local == start_time_bulk  # No offset here

            duration = a.get("duration")
            # print(d["Signal"])
            len_signal = len(d["Signal"][()])
            assert len_signal == duration

            voltage = a.get("voltage")
            assert voltage == config["segment"]["voltage_threshold"]

    assert n_captures == actual_n_captures
    os.remove(capture_f5_fname)


def sort_capture_windows_by_channel_test():
    signal_metadata = np.array(
        [
            # channel_no, capture_window, offset, rng, digi
            [1, (0, 20000), 0, 0, 0],
            [2, (100001, 200001), 0, 0, 0],
            [1, (100000, 200000), 0, 0, 0],
            [1, (50000, 80000), 0, 0, 0],
            [3, (1234, 10000), 0, 0, 0],
            [2, (1111, 20000), 0, 0, 0],
        ],
        dtype=object,
    )
    sorted = segment.sort_capture_windows_by_channel(signal_metadata)
    valid_channels = [1, 2, 3]
    valid_counts = [3, 2, 1]
    for channel, count in zip(valid_channels, valid_counts):
        meta = sorted.get(channel)
        assert len(meta) == count
        last_window = meta[0]
        for window in meta[1:]:
            assert window[0] >= last_window[0]
            last_window = window


def write_capture_windows_to_fast5_test():
    capture_f5_fname = "tests/write_windows_test_dummy.fast5"
    if os.path.exists(capture_f5_fname):
        os.remove(capture_f5_fname)
    signal_metadata = np.array(
        [
            # channel_no, capture_window, offset, rng, digi
            [1, (0, 20000), 0, 0, 0],
            [2, (100001, 200001), 0, 0, 0],
            [1, (100000, 200000), 0, 0, 0],
            [1, (50000, 80000), 0, 0, 0],
            [3, (1234, 10000), 0, 0, 0],
            [2, (1111, 20000), 0, 0, 0],
        ],
        dtype=object,
    )
    segment.write_capture_windows_to_fast5(capture_f5_fname, signal_metadata)
    assert os.path.exists(capture_f5_fname)
    os.remove(capture_f5_fname)


def write_capture_to_fast5_test(tmpdir):
    capture_f5_fname = "tests/write_test_dummy.fast5"
    if os.path.exists(capture_f5_fname):
        os.remove(capture_f5_fname)
    read_id = "1405caa5-74fd-4478-8fac-1d0b5d6ead8e"
    signal_pA = np.random.rand(5000)
    start_time_bulk = 10000
    start_time_local = 0
    duration = 8000
    voltage = -180
    open_channel_pA = 229.1
    ejected = True
    channel_no = 3
    offset, rng, digi = -21.0, 3013.53, 8192.0
    sampling_rate = 10000
    segment.write_capture_to_fast5(
        capture_f5_fname,
        read_id,
        signal_pA,
        start_time_bulk,
        start_time_local,
        duration,
        ejected,
        voltage,
        open_channel_pA,
        channel_no,
        digi,
        offset,
        rng,
        sampling_rate,
    )
    assert os.path.exists(capture_f5_fname)
    # TODO further validation, incl. contents of file
    os.remove(capture_f5_fname)


# def add_capture_windows_test():
#     # Hack to add capture windows to test data
#     dir = "tests/data"
#     bulk_f5_fname = "tests/data/bulk_fast5_dummy.fast5"
#     files = [
#         os.path.join(dir, x)
#         for x in os.listdir(dir)
#         if "bulk" not in x and x.endswith("fast5")
#     ]
#     for f in files:
#         raw_signals, context, run_id, sampling_rate = segment._prep_capture_windows(
#             bulk_f5_fname, None, None, -180, 0.7, [1, 2, 3], 229.1,
#         )
#        segment.write_capture_windows_to_fast5(f, context)
