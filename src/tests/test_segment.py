"""
================
test_segment.py
================

This module contains tests for segment.py functionality.

"""
import os
from logging import debug
from pathlib import Path

import h5py
import numpy as np
import pytest
from pytest import fixture

from src.poretitioner.logger import configure_root_logger, getLogger, verbosity_to_log_level
from src.poretitioner.signals import (
    CaptureMetadata,
    ChannelCalibration,
    PicoampereSignal,
    RawSignal,
    Window,
)
from src.poretitioner.utils import filtering, segment
from src.poretitioner.utils.configuration import GeneralConfiguration, SegmentConfiguration

configure_root_logger(verbosity=1, debug=False)


@fixture
def get_data(tmp_path, request):
    test_dir = tmp_path
    data_dir = Path(test_dir, "data")
    print(f"\n\n\nTEST DATA DIR: {data_dir!s}\n\n\n")


@fixture
def old_config():
    segment_config = {
        "voltage_threshold": -140,
        "signal_threshold": 0.7,
        "translocation_delay": 20,
        "open_channel_prior_mean": 220,
        "good_channels": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "end_tol": 50,
        "terminal_capture_only": True,
        "filter": {"length": (100, None)},
    }
    config = {"segment": segment_config}

    yield config


# def create_capture_fast5_test(get_data, old_config):
#     # TODO deprecated function, remove
#     """Test valid capture fast5 produced. Conditions:
#     * Valid bulk fast5 input
#     * Valid capture fast5 path
#     * Empty segmenter config dict (valid)
#     * No sub runs
#     """
#     bulk_f5_fname = "src/tests/data/bulk_fast5_dummy.fast5"
#     capture_f5_fname = "src/tests/data/capture_fast5_dummy.fast5"
#     segment.create_capture_fast5(
#         bulk_f5_fname, capture_f5_fname, old_config, overwrite=True, sub_run=None
#     )
#     assert os.path.exists(capture_f5_fname)
#     with h5py.File(capture_f5_fname, "r") as f5:
#         assert f5["Meta/context_tags"] is not None
#         assert "bulk_filename" in list(f5["Meta/context_tags"].attrs)
#         assert f5["Meta/tracking_id"] is not None
#         assert "sub_run_id" not in list(f5["Meta/tracking_id"].attrs)
#         assert "sub_run_offset" not in list(f5["Meta/tracking_id"].attrs)
#         assert "sub_run_duration" not in list(f5["Meta/tracking_id"].attrs)
#     os.remove(capture_f5_fname)


# def create_capture_fast5_bulk_exists_test():
#     """Test error thrown when bulk fast5 does not exist."""
#     bulk_f5_fname = "src/tests/data/bulk_fast5_dummy_fake.fast5"
#     capture_f5_fname = "src/tests/data/capture_fast5_dummy_bulkdemo.fast5"
#     config = {}
#     with pytest.raises(OSError):
#         segment.create_capture_fast5(
#             bulk_f5_fname, capture_f5_fname, config, overwrite=False, sub_run=None
#         )


# def create_capture_fast5_capture_path_missing_test():
#     """Test error thrown when bulk fast5 does not exist."""
#     bulk_f5_fname = "src/tests/data/bulk_fast5_dummy_fake.fast5"
#     capture_f5_fname = "src/tests/DNE/capture_fast5_dummy_bulkdemo.fast5"
#     config = {}
#     with pytest.raises(OSError):
#         segment.create_capture_fast5(
#             bulk_f5_fname, capture_f5_fname, config, overwrite=False, sub_run=None
#         )


# def create_capture_fast5_subrun_test():
#     """Test valid capture fast5 produced. Conditions:
#     * Valid bulk fast5 input
#     * Valid capture fast5 path
#     * Empty segmenter config dict (valid)
#     * HAS sub run
#     """
#     bulk_f5_fname = "src/tests/data/bulk_fast5_dummy.fast5"
#     capture_f5_fname = "src/tests/data/capture_fast5_dummy_sub.fast5"
#     segment_config = {
#         "voltage_threshold": -140,
#         "signal_threshold": 0.7,
#         "translocation_delay": 20,
#         "open_channel_prior_mean": 220,
#         "good_channels": [2, 3, 4, 5, 6, 7, 8, 9, 10],
#         "end_tol": 50,
#         "terminal_capture_only": True,
#         "filter": {"length": (100, None)},
#     }
#     config = {"segment": segment_config}
#     segment.create_capture_fast5(
#         bulk_f5_fname,
#         capture_f5_fname,
#         config,
#         overwrite=True,
#         sub_run=("S8w9g33", 9_999_999, 1_000_000),
#     )
#     with h5py.File(capture_f5_fname, "r") as f5:
#         assert "sub_run_id" in list(f5["Meta/tracking_id"].attrs)
#         assert "S8w9g33" in str(f5["Meta/tracking_id"].attrs["sub_run_id"])
#         assert "sub_run_offset" in list(f5["Meta/tracking_id"].attrs)
#         assert f5["Meta/tracking_id"].attrs["sub_run_offset"] == 9_999_999
#         assert "sub_run_duration" in list(f5["Meta/tracking_id"].attrs)
#         assert f5["Meta/tracking_id"].attrs["sub_run_duration"] == 1_000_000
#     os.remove(capture_f5_fname)


def picoampere_signal_from_data_file(
    filepath: str,
    channel_number=1,
    calibration: ChannelCalibration = ChannelCalibration(0, 1, 1),
) -> PicoampereSignal:
    """Helper method to extract a PicoampereSignal from a raw data file.

    Parameters
    ----------
    filepath : str
        Path to a signal file.
    channel_number : int, optional
        Channel index, by default 1
    calibration : ChannelCalibraticlearon, optional
        How to convert this raw signal to picoamperes, by default ChannelCalibration(0, 1, 1) (i.e. the identity calibration: just multiplies everything 1)

    Returns
    -------
    PicoampereSignal
        Signal in picoamperes.
    """
    data = np.loadtxt(filepath, delimiter="\t", comments="#")
    raw = RawSignal(data, channel_number, calibration)
    pico = raw.to_picoamperes()
    return pico


def find_captures_0_single_capture_test():
    data_file = "src/tests/data/capture_windows/test_data_capture_window_0.txt.gz"
    window = Window(3_572_989, 3_665_680)
    data = picoampere_signal_from_data_file(data_file)
    actual_captures = [(33832 + window.start, 92691 + window.start, True)]
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = [filtering.LengthFilter(100, None)]
    delay = 10
    end_tol = 0
    channel_number = 1
    captures = segment.find_captures(
        data,
        channel_number,
        window,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == len(actual_captures)
    for test_capture in captures:
        test_start = test_capture.window.start
        test_end = test_capture.window.end
        ejected = test_capture.ejected
        assert (test_start, test_end, ejected) in actual_captures


def find_captures_0_single_capture_terminal_test():
    data_file = "src/tests/data/capture_windows/test_data_capture_window_0.txt.gz"
    window = Window(3_572_989, 3_665_680)
    data = picoampere_signal_from_data_file(data_file)
    actual_captures = [(33822 + window.start, 92691 + window.start, True)]
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = True
    filters = [filtering.LengthFilter(100, None)]
    delay = 0
    end_tol = 0
    channel_number = 1
    captures = segment.find_captures(
        data,
        channel_number,
        window,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )

    assert len(captures) == len(actual_captures)
    for test_capture in captures:
        test_start = test_capture.window.start
        test_end = test_capture.window.end
        ejected = test_capture.ejected
        assert (test_start, test_end, ejected) in actual_captures


def find_captures_1_double_capture_noterminal_test():
    """Example capture window contains 2 long captures, neither terminal.
    Also contains a few short blips.

    Test: terminal_capture_only = True returns no captures"""
    data_file = "src/tests/data/capture_windows/test_data_capture_window_1.txt.gz"
    data = picoampere_signal_from_data_file(data_file)
    window = Window(4_765_695, 4_858_482)
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = True
    filters = [filtering.LengthFilter(100, None)]
    delay = 0
    end_tol = 0
    channel_number = 1
    captures = segment.find_captures(
        data,
        channel_number,
        window,
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
    data_file = "src/tests/data/capture_windows/test_data_capture_window_1.txt.gz"
    data = picoampere_signal_from_data_file(data_file)
    window = Window(4_765_695, 4_858_482)
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = [filtering.LengthFilter(100, None)]
    delay = 0
    end_tol = 0
    channel_number = 1
    captures = segment.find_captures(
        data,
        channel_number,
        window,
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
    data_file = "src/tests/data/capture_windows/test_data_capture_window_2.txt.gz"
    data = picoampere_signal_from_data_file(data_file)
    window = Window(3_423_474, 3_516_439)
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = [filtering.LengthFilter(100, None)]
    delay = 10
    end_tol = 0
    channel_number = 1
    captures = segment.find_captures(
        data,
        channel_number,
        window,
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
    data_file = "src/tests/data/capture_windows/test_data_capture_window_3.txt.gz"
    data = picoampere_signal_from_data_file(data_file)
    window = Window(1_187_841, 1_280_674)
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = True
    filters = [filtering.LengthFilter(100, None)]
    delay = 0
    end_tol = 0
    channel_number = 2
    captures = segment.find_captures(
        data,
        channel_number,
        window,
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
    data_file = "src/tests/data/capture_windows/test_data_capture_window_3.txt.gz"
    data = picoampere_signal_from_data_file(data_file)
    window = Window(1_187_841, 1_280_674)
    actual_captures = [(1_200_088, 1_201_033, False), (1_252_611, 1_280_674, True)]
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = [filtering.LengthFilter(100, None)]
    delay = 3
    end_tol = 0
    channel_number = 2
    captures = segment.find_captures(
        data,
        channel_number,
        window,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == len(actual_captures)
    for test_capture in captures:
        test_start = test_capture.window.start
        test_end = test_capture.window.end
        ejected = test_capture.ejected
        assert (test_start, test_end, ejected) in actual_captures


def find_captures_4_unfolded_terminal_test():
    """Example capture window contains 1 long terminal capturethat has unfolded.

    Tests: find_captures returns 1 capture
    """
    data_file = "src/tests/data/capture_windows/test_data_capture_window_4.txt.gz"
    data = picoampere_signal_from_data_file(data_file)
    window = Window(5_511_887, 5_604_585)
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = True
    filters = [filtering.LengthFilter(100, None)]
    delay = 0
    end_tol = 0
    channel_number = 2
    captures = segment.find_captures(
        data,
        channel_number,
        window,
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
    data_file = "src/tests/data/capture_windows/test_data_capture_window_5.txt.gz"
    data = picoampere_signal_from_data_file(data_file)
    window = Window(965_676, 1_059_216)
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = True
    filters = [filtering.LengthFilter(100, None)]
    delay = 0
    end_tol = 0
    channel_number = 1
    captures = segment.find_captures(
        data,
        channel_number,
        window,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 1
    open_channel_pA = np.array([capture.open_channel_pA_calculated for capture in captures])
    low_expected_open_channel_pA = 228.5
    high_expected_open_channel_pA = 230
    # Rough check; should be ~229.05 & anything close is okay.
    # The function is nondeterministic & should return this exact value, but if
    # future changes are made, some tolerance can be allowed.
    all_currents_within_bounds = all(
        (open_channel_pA > low_expected_open_channel_pA)
        & (open_channel_pA < high_expected_open_channel_pA)
    )
    assert (
        all_currents_within_bounds
    ), f"Expect all capture open channel currents to be between '{low_expected_open_channel_pA}' and '{high_expected_open_channel_pA}'."


def find_captures_6_clog_no_open_channel_test():
    """Example capture window contains 1 long terminal capture. Open pore region
    is extremely, extremely short. Test by cutting off the open pore region.

    Tests: find_captures returns 1 capture; open pore returns alt value.
    """
    data_file = "src/tests/data/capture_windows/test_data_capture_window_6.txt.gz"
    data = picoampere_signal_from_data_file(data_file)[100:]
    window = Window(2_769_436, 2_863_265)
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = [filtering.LengthFilter(100, None)]
    delay = 100
    end_tol = 0
    channel_number = 1
    captures = segment.find_captures(
        data,
        channel_number,
        window,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 1
    open_channel_pA = np.array([capture.open_channel_pA_calculated for capture in captures])
    expected_open_channel_pA = 230
    all_currents_within_bounds = all(
        (np.isclose(open_channel_pA, expected_open_channel_pA, atol=0.5))
    )

    assert (
        all_currents_within_bounds
    ), f"All calculated open channel currents should be close to {expected_open_channel_pA}"


def find_captures_7_capture_no_open_channel_test():
    """Example capture window contains 1 long terminal capture. Open pore region
    is extremely, extremely short. Test by cutting off the open pore region.

    Tests: find_captures returns 1 capture; open pore returns alt value.
    """
    data_file = "src/tests/data/capture_windows/test_data_capture_window_7.txt.gz"
    data = picoampere_signal_from_data_file(data_file)[100:]
    window = Window(2_919_913, 3_013_723)
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = [filtering.LengthFilter(100, None)]
    delay = 100
    end_tol = 0
    channel_number = 2
    captures = segment.find_captures(
        data,
        channel_number,
        window,
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
    expected_open_channel_pA = 230

    open_channel_pA = np.array([capture.open_channel_pA_calculated for capture in captures])
    all_currents_within_bounds = all(
        (np.isclose(open_channel_pA, expected_open_channel_pA, atol=0.5))
    )

    assert (
        all_currents_within_bounds
    ), f"All captures should have calculated an open channel current close to {expected_open_channel_pA}."


def find_captures_8_capture_no_open_channel_test():
    """Example capture window contains 2 captures: both long, 1 terminal.
    Test non-terminal long capture.

    Tests: find_captures returns 2 captures.
           Checks exact capture boundaries with delay = 3
    """
    data_file = "src/tests/data/capture_windows/test_data_capture_window_8.txt.gz"
    data = picoampere_signal_from_data_file(data_file)
    window = Window(4_875_289, 4_969_337)
    signal_threshold_frac = 0.7
    alt_open_channel_pA = 230
    terminal_capture_only = False
    filters = [filtering.LengthFilter(100, None)]
    delay = 3
    end_tol = 0
    channel_number = 2
    captures = segment.find_captures(
        data,
        channel_number,
        window,
        signal_threshold_frac,
        alt_open_channel_pA,
        terminal_capture_only=terminal_capture_only,
        filters=filters,
        delay=delay,
        end_tol=end_tol,
    )
    assert len(captures) == 2
    actual_captures = [
        (11310 + window.start, 22098 + window.start, False),
        (26617 + window.start, 94048 + window.start, True),
    ]
    for test_capture in captures:
        test_start = test_capture.window.start
        test_end = test_capture.window.end
        ejected = test_capture.ejected
        assert (test_start, test_end, ejected) in actual_captures


def generate_read_id_test():
    read_id = segment.generate_read_id()
    assert type(read_id) is str
    assert len(read_id) == 36


def prep_capture_windows_test():
    bulk_f5_fname = "src/tests/data/bulk_fast5_dummy.fast5"
    voltage_threshold = -180
    signal_threshold_frac = 0.7
    good_channels = [1, 2, 3]
    open_channel_pA_prior = 220
    open_channel_pA_prior_bound = 40
    prepped = segment._prep_capture_windows(
        bulk_f5_fname,
        voltage_threshold,
        signal_threshold_frac,
        good_channels,
        open_channel_pA_prior,
        open_channel_pA_prior_bound,
    )
    count_by_channel = {1: 0, 2: 0, 3: 0}
    for cap in prepped.captures:
        count_by_channel[cap.signal.channel_number] += 1
    for channel_number in good_channels:
        assert count_by_channel[channel_number] == 4


@pytest.mark.xfail(reason="Need to implement config (filters currently in progress).")
class TestParallelFindCaptures:
    def parallel_find_captures_test(self):
        bulk_f5_fname = "src/tests/data/bulk_fast5_dummy.fast5"

        filters = [filtering.LengthFilter(100, None)]
        config = GeneralConfiguration(config={"n_workers": 2, "capture_directory": "src/tests"})

        segment_config = {
            "voltage_threshold": -180,
            "signal_threshold_frac": 0.7,
            "translocation_delay": 20,
            "open_channel_prior_mean": 220,
            "open_channel_prior_stdv": 50,
            "good_channels": [1, 3],
            "end_tolerance": 50,
            "terminal_capture_only": False,
            "n_captures_per_file": 1000,
            "bulkfast5": bulk_f5_fname,
        }
        segment_config = SegmentConfiguration(segment_config)

        segment.parallel_find_captures(config, segment_config, overwrite=True, filters=filters)

        run_id = "d0befb838f5a9a966e3c559dc3a75a6612745849"
        actual_n_captures = 5
        n_captures = 0
        capture_f5_fname = f"src/tests/{run_id}_1.fast5"
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
                len_signal = len(d["Signal"][()])
                assert len_signal == duration

                voltage = a.get("voltage")
                assert voltage == segment_config.voltage_threshold
                print(duration, a.get("channel_number"))
        assert n_captures == actual_n_captures
        os.remove(capture_f5_fname)

    def parallel_find_captures_overflow_file_test(self):
        bulk_f5_fname = "src/tests/data/bulk_fast5_dummy.fast5"

        filters = [filtering.LengthFilter(100, None)]
        config = GeneralConfiguration(config={"n_workers": 2, "capture_directory": "src/tests"})

        segment_config = {
            "voltage_threshold": -180,
            "signal_threshold_frac": 0.7,
            "translocation_delay": 20,
            "open_channel_prior_mean": 220,
            "open_channel_prior_stdv": 50,
            "good_channels": [1, 3],
            "end_tolerance": 50,
            "terminal_capture_only": False,
            "n_captures_per_file": 2,
            "bulkfast5": bulk_f5_fname,
        }
        segment_config = SegmentConfiguration(segment_config)

        segment.parallel_find_captures(config, segment_config, overwrite=True, filters=filters)
        run_id = "d0befb838f5a9a966e3c559dc3a75a6612745849"
        actual_n_captures = 5
        n_captures = 0
        capture_f5_fnames = [
            os.path.join("src/tests/", x) for x in os.listdir("src/tests/") if run_id in x
        ]
        assert len(capture_f5_fnames) == 3
        for capture_f5_fname in capture_f5_fnames:
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
                    len_signal = len(d["Signal"][()])
                    assert len_signal == duration

                    voltage = a.get("voltage")
                    assert voltage == segment_config.voltage_threshold
                    print(duration, a.get("channel_number"))
            os.remove(capture_f5_fname)
        assert n_captures == actual_n_captures


class TestSegment:
    def segment_test(self):
        bulk_f5_fname = "src/tests/data/bulk_fast5_dummy.fast5"

        filters = [filtering.LengthFilter(100, None)]
        config = GeneralConfiguration(config={"n_workers": 2, "capture_directory": "src/tests"})

        segment_config = {
            "voltage_threshold": -180,
            "signal_threshold_frac": 0.7,
            "translocation_delay": 20,
            "open_channel_prior_mean": 220,
            "open_channel_prior_stdv": 50,
            "good_channels": [
                1,
                2,
                3,
            ],  # this will be internally overwritten by the good channels calculation, which should not include channel 2
            "end_tolerance": 50,
            "terminal_capture_only": False,
            "n_captures_per_file": 1000,
            "bulkfast5": bulk_f5_fname,
        }
        segment_config = SegmentConfiguration(segment_config)

        segment.segment(bulk_f5_fname, config, segment_config, overwrite=True, filters=filters)
        run_id = "d0befb838f5a9a966e3c559dc3a75a6612745849"
        actual_n_captures = 5
        n_captures = 0
        capture_f5_fname = f"src/tests/{run_id}_1.fast5"
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
                len_signal = len(d["Signal"][()])
                assert len_signal == duration

                voltage = a.get("voltage")
                assert voltage == segment_config.voltage_threshold
                print(duration, a.get("channel_number"))
        assert n_captures == actual_n_captures
        os.remove(capture_f5_fname)
