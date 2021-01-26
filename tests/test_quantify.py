"""
================
test_quantify.py
================

This module contains tests for quantify.py functionality.

"""
import h5py
import numpy as np

import poretitioner.utils.quantify as quantify


def calc_time_until_capture_test():
    # Tests the quantification method used in the paper
    # I know this hardcoding isn't ideal
    capture_windows = [
        (93670, 273393),
        (329867, 422733),
        (479354, 572158),
        (628692, 721058),
        (777500, 870168),
        (926761, 1019080),
        (1075696, 1168519),
        (1225213, 1317929),
        (1374460, 1466794),
        (1523344, 1615653),
        (1672162, 1764825),
        (1824809, 1913639),
        (1970238, 2062795),
        (2119442, 2211806),
        (2268328, 2360504),
        (2417001, 2509470),
        (2565915, 2658269),
        (2714729, 2808289),
        (2865049, 2956785),
        (3013459, 3106239),
        (3162741, 3255225),
        (3311741, 3404420),
        (3460828, 3553166),
    ]
    captures = [(986724, 1019080), (3342036, 3404420)]
    blockages = [
        (128050, 128263),
        (128648, 128663),
        (187295, 188662),
        (200386, 273393),
        (482438, 572158),
        (643345, 644207),
        (865991, 870168),
        (986724, 1019080),
        (1137563, 1168519),
        (1293873, 1317929),
        (1411160, 1466794),
        (1674245, 1764825),
        (1905230, 1905240),
        (1906834, 1906909),
        (1980012, 2062795),
        (2140753, 2211806),
        (2292331, 2292466),
        (2329304, 2360504),
        (2417257, 2417269),
        (2446937, 2509470),
        (2623543, 2658269),
        (2727114, 2771431),
        (2794550, 2808289),
        (3051667, 3057059),
        (3080848, 3106239),
        (3171765, 3200000),
        (3342036, 3404420),
        (3546470, 3553166),
    ]
    original_output = [441029, 813623]
    tested_output = quantify.calc_time_until_capture(
        capture_windows, captures, blockages=blockages
    )
    assert len(original_output) == len(tested_output)
    for i in range(len(original_output)):
        assert original_output[i] == tested_output[i]


def get_capture_details_in_f5_test():
    fast5_fname = "tests/data/classified_9captures.fast5"
    with h5py.File(fast5_fname, "r") as f5:
        results = quantify.get_capture_details_in_f5(f5)
        assert len(results) == 9
    # Note that the following test assumes a fixed order for the returned values.
    # This may not be guaranteed in the future.
    check_values = ["697de4c1-1aef-41b9-ae0d-d676e983cb7e", 201392, 285550, 1, None]
    for returned, check in zip(results[0], check_values):
        assert returned == check


def get_capture_windows_by_channel_test():
    fast5_fname = "tests/data/classified_9captures.fast5"
    windows_by_channel = quantify.get_capture_windows_by_channel(fast5_fname)
    valid_channels = [1, 2, 3]
    valid_counts = [4, 4, 4]
    for channel, count in zip(valid_channels, valid_counts):
        meta = windows_by_channel.get(channel)
        assert len(meta) == count
        last_window = meta[0]
        for window in meta[1:]:
            assert window[0] >= last_window[0]
            last_window = window


def sort_captures_by_channel_test():
    captures = np.array(
        [
            ["697de4c1-1aef-41b9-ae0d-d676e983cb7e", 201392, 285550, 1, None],
            ["8e8181d2-d749-4735-9cab-37648b463f88", 367644, 435768, 3, None],
            ["97d06f2e-90e6-4ca7-91c4-084f13b693f2", 41680, 135380, 2, None],
            ["a0c40f5a-c685-43b9-a3b7-ca13aa90d832", 492342, 500000, 3, None],
            ["ab54dab5-26a7-4062-9d77-f63fc40f702c", 492342, 500000, 2, None],
            ["c87905e6-fd62-4ac6-bcbd-c7f17ff4af14", 192049, 285550, 2, None],
            ["cd6fa746-e93b-467f-a3fc-1c9af815f836", 52872, 135380, 3, None],
            ["df4365f4-bfe4-4d2c-8100-34c36cd11378", 342119, 435768, 2, None],
            ["f5d76520-c92b-4a9c-b5cb-a04414db527e", 370535, 435768, 1, None],
        ]
    )
    captures_by_channel = quantify.sort_captures_by_channel(captures)
    valid_channels = [1, 2, 3]
    n_captures = [None, 2, 4, 3]
    for channel, captures in captures_by_channel.items():
        # Check for expected channels & #/captures per channel
        assert channel in valid_channels
        assert n_captures[channel] == len(captures)
        # Check ordering of captures (earliest to latest)
        last_capture = captures[0]
        for capture in captures[1:]:
            assert capture[1] >= last_capture[1]
            last_capture = capture


def quantify_files_time_until_capture_test():
    # Define config dict that contains filter info
    config = {
        "compute": {"n_workers": 4},
        "filters": {
            "base filter": {"length": (100, None)},
            "test filter": {"min": (100, None)},
        },
        "output": {"capture_f5_dir": "tests/", "captures_per_f5": 1000},
        "classify": {
            "classifier": "NTER_cnn",
            "classifier_path": "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt",
            "classifier_version": "1.0",
            "start_obs": 100,
            "end_obs": 21000,
            "min_confidence": 0.9,
        },
    }
    fast5_fnames = ["tests/data/classified_9captures.fast5"]
    overwrite = False
    filter_name = None
    quant_method = "time between captures"
    times = quantify.quantify_files(
        config,
        fast5_fnames,
        overwrite=overwrite,
        filter_name=filter_name,
        quant_method=quant_method,
        classified_only=True,
    )
    assert len(times) == 1
    assert times[0] == 93720
