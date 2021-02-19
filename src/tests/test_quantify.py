"""
================
test_quantify.py
================

This module contains tests for quantify.py functionality.

"""
import h5py
import numpy as np
import poretitioner.utils.quantify as quantify
from poretitioner.utils.core import Window


def calc_time_until_capture_test():
    # Tests the quantification method used in the paper
    # I know this hardcoding isn't ideal
    capture_windows = [
        Window(93670, 273393),
        Window(329867, 422733),
        Window(479354, 572158),
        Window(628692, 721058),
        Window(777500, 870168),
        Window(926761, 1019080),
        Window(1075696, 1168519),
        Window(1225213, 1317929),
        Window(1374460, 1466794),
        Window(1523344, 1615653),
        Window(1672162, 1764825),
        Window(1824809, 1913639),
        Window(1970238, 2062795),
        Window(2119442, 2211806),
        Window(2268328, 2360504),
        Window(2417001, 2509470),
        Window(2565915, 2658269),
        Window(2714729, 2808289),
        Window(2865049, 2956785),
        Window(3013459, 3106239),
        Window(3162741, 3255225),
        Window(3311741, 3404420),
        Window(3460828, 3553166),
    ]
    captures = [Window(986724, 1019080), Window(3342036, 3404420)]
    blockages = [
        Window(128050, 128263),
        Window(128648, 128663),
        Window(187295, 188662),
        Window(200386, 273393),
        Window(482438, 572158),
        Window(643345, 644207),
        Window(865991, 870168),
        Window(986724, 1019080),
        Window(1137563, 1168519),
        Window(1293873, 1317929),
        Window(1411160, 1466794),
        Window(1674245, 1764825),
        Window(1905230, 1905240),
        Window(1906834, 1906909),
        Window(1980012, 2062795),
        Window(2140753, 2211806),
        Window(2292331, 2292466),
        Window(2329304, 2360504),
        Window(2417257, 2417269),
        Window(2446937, 2509470),
        Window(2623543, 2658269),
        Window(2727114, 2771431),
        Window(2794550, 2808289),
        Window(3051667, 3057059),
        Window(3080848, 3106239),
        Window(3171765, 3200000),
        Window(3342036, 3404420),
        Window(3546470, 3553166),
    ]
    original_output = [441029, 813623]
    tested_output = quantify.calc_time_until_capture(
        capture_windows, captures, blockages=blockages
    )
    assert len(original_output) == len(tested_output)
    for i in range(len(original_output)):
        assert original_output[i] == tested_output[i]


# TODO: Restore quantification tests: https://github.com/uwmisl/poretitioner/issues/89
# def get_capture_details_in_f5_test():
#     fast5_fname = "src/tests/data/classified_9captures.fast5"
#     with h5py.File(fast5_fname, "r") as f5:
#         results = quantify.get_capture_details_in_f5(f5)
#         assert len(results) == 9
#     # Note that the following test assumes a fixed order for the returned values.
#     # This may not be guaranteed in the future.
#     check_values = ["697de4c1-1aef-41b9-ae0d-d676e983cb7e", 201392, 285550, 1, None]
#     for returned, check in zip(results[0], check_values):
#         assert returned == check


# def get_capture_windows_by_channel_test():
#     fast5_fname = "src/tests/data/classified_9captures.fast5"
#     windows_by_channel = quantify.get_capture_windows_by_channel(fast5_fname)
#     valid_channels = [1, 2, 3]
#     valid_counts = [4, 4, 4]
#     for channel, count in zip(valid_channels, valid_counts):
#         meta = windows_by_channel.get(channel)
#         assert len(meta) == count
#         last_window = meta[0]
#         for window in meta[1:]:
#             assert window[0] >= last_window[0]
#             last_window = window


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


# def quantify_files_time_until_capture_test():
#     # Define config dict that contains filter info
#     config = {
#         "compute": {"n_workers": 4},
#         "filters": {"base filter": {"length": (100, None)}, "test filter": {"min": (100, None)}},
#         "output": {"capture_f5_dir": "src/tests/", "captures_per_f5": 1000},
#         "classify": {
#             "classifier": "NTER_cnn",
#             "classifier_path": "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt",
#             "classifier_version": "1.0",
#             "start_obs": 100,
#             "end_obs": 21000,
#             "min_confidence": 0.9,
#         },
#     }
#     fast5_fnames = ["src/tests/data/classified_9captures.fast5"]
#     filter_name = None
#     quant_method = "time_between_captures"
#     times = quantify.quantify_files(
#         config,
#         fast5_fnames,
#         filter_name=filter_name,
#         quant_method=quant_method,
#         classified_only=True,
#     )
#     assert len(times) == 1
#     assert times[0] == 93720


# def quantify_files_time_until_capture_intervals_test():
#     config = {
#         "compute": {"n_workers": 4},
#         "filters": {"base filter": {"length": (100, None)}, "test filter": {"min": (100, None)}},
#         "output": {"capture_f5_dir": "src/tests/", "captures_per_f5": 1000},
#         "classify": {
#             "classifier": "NTER_cnn",
#             "classifier_path": "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt",
#             "classifier_version": "1.0",
#             "start_obs": 100,
#             "end_obs": 21000,
#             "min_confidence": 0.9,
#         },
#     }
#     fast5_fnames = ["src/tests/data/classified_10mins_4channels.fast5"]
#     filter_name = None
#     quant_method = "time_between_captures"
#     interval_mins = 3
#     times = quantify.quantify_files(
#         config,
#         fast5_fnames,
#         filter_name=filter_name,
#         quant_method=quant_method,
#         classified_only=True,
#         interval_mins=interval_mins,
#     )
#     assert len(times) == 4
#     assert times[0] < 220401 and times[0] > 220400


# def quantify_files_capture_freq_test():
#     config = {
#         "compute": {"n_workers": 4},
#         "filters": {"base filter": {"length": (100, None)}, "test filter": {"min": (100, None)}},
#         "output": {"capture_f5_dir": "src/tests/", "captures_per_f5": 1000},
#         "classify": {
#             "classifier": "NTER_cnn",
#             "classifier_path": "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt",
#             "classifier_version": "1.0",
#             "start_obs": 100,
#             "end_obs": 21000,
#             "min_confidence": 0.9,
#         },
#     }
#     fast5_fnames = ["src/tests/data/classified_9captures.fast5"]
#     filter_name = None
#     quant_method = "capture_freq"
#     times = quantify.quantify_files(
#         config,
#         fast5_fnames,
#         filter_name=filter_name,
#         quant_method=quant_method,
#         classified_only=True,
#     )
#     assert len(times) == 1
#     assert times[0] < 2.08 and times[0] > 2.07


# def quantify_files_capture_freq_intervals_test():
#     config = {
#         "compute": {"n_workers": 4},
#         "filters": {"base filter": {"length": (100, None)}, "test filter": {"min": (100, None)}},
#         "output": {"capture_f5_dir": "src/tests/", "captures_per_f5": 1000},
#         "classify": {
#             "classifier": "NTER_cnn",
#             "classifier_path": "poretitioner/utils/model/NTERs_trained_cnn_05152019.statedict.pt",
#             "classifier_version": "1.0",
#             "start_obs": 100,
#             "end_obs": 21000,
#             "min_confidence": 0.9,
#         },
#     }
#     fast5_fnames = ["src/tests/data/classified_10mins_4channels.fast5"]
#     filter_name = None
#     quant_method = "capture_freq"
#     interval_mins = 3
#     times = quantify.quantify_files(
#         config,
#         fast5_fnames,
#         filter_name=filter_name,
#         quant_method=quant_method,
#         classified_only=True,
#         interval_mins=interval_mins,
#     )
#     assert len(times) == 4
#     assert times[0] < 1.17 and times[0] > 1.16
