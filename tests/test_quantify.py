"""
================
test_quantify.py
================

This module contains tests for quantify.py functionality.

"""
import poretitioner.utils.quantify as quantify


def calc_time_until_capture_test():
    # I know this hardcoding isn't ideal
    capture_windows = [(93670, 273393), (329867, 422733), (479354, 572158),
                       (628692, 721058), (777500, 870168), (926761, 1019080),
                       (1075696, 1168519), (1225213, 1317929), (1374460, 1466794),
                       (1523344, 1615653), (1672162, 1764825), (1824809, 1913639),
                       (1970238, 2062795), (2119442, 2211806), (2268328, 2360504),
                       (2417001, 2509470), (2565915, 2658269), (2714729, 2808289),
                       (2865049, 2956785), (3013459, 3106239), (3162741, 3255225),
                       (3311741, 3404420), (3460828, 3553166)]
    captures = [(986724, 1019080), (3342036, 3404420)]
    blockages = [(128050, 128263), (128648, 128663), (187295, 188662),
                 (200386, 273393), (482438, 572158), (643345, 644207),
                 (865991, 870168), (986724, 1019080), (1137563, 1168519),
                 (1293873, 1317929), (1411160, 1466794), (1674245, 1764825),
                 (1905230, 1905240), (1906834, 1906909), (1980012, 2062795),
                 (2140753, 2211806), (2292331, 2292466), (2329304, 2360504),
                 (2417257, 2417269), (2446937, 2509470), (2623543, 2658269),
                 (2727114, 2771431), (2794550, 2808289), (3051667, 3057059),
                 (3080848, 3106239), (3171765, 3200000), (3342036, 3404420),
                 (3546470, 3553166)]
    original_output = [441029, 813623]
    tested_output = quantify.calc_time_until_capture(capture_windows, captures,
                                                     blockages=blockages)
    assert len(original_output) == len(tested_output)
    for i in range(len(original_output)):
        assert original_output[i] == tested_output[i]


def get_overlapping_regions_test():
    window = (10, 100)
    excl_regions = [(0, 9), (9, 10), (100, 101), (1000, 1001)]
    overlap = quantify.get_overlapping_regions(window, excl_regions)
    assert len(overlap) == 0
    incl_regions = [(9, 11), (20, 40), (99, 100), (99, 1000)]
    overlap = quantify.get_overlapping_regions(window, incl_regions)
    assert len(overlap) == len(incl_regions)
