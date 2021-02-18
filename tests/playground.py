"""
===================
playground.py
===================

Makes it easy to test small snippets of code, testing how various app
components work together.
"""

import pprint
import sys

import numpy as np
from poretitioner.application_info import *
from poretitioner.fast5s import *
from poretitioner.getargs import *
from poretitioner.logger import *
from poretitioner.signals import *
from poretitioner.utils import *

CALIBRATION = ChannelCalibration(0, 2, 1)
CHANNEL_NUMBER = 1
OPEN_CHANNEL_GUESS = 45
OPEN_CHANNEL_BOUND = 10
RANDOM_SIGNAL = np.random.randn(100)
BULK_FAST5_FILE = "../tests/data/bulk_fast5_dummy.fast5"
CAPTURE_FAST5_FILE = "../tests/data/reads_fast5_dummy_9captures.fast5"
CLASSIFIED_FAST5_FILE = "../tests/data/classified_9captures.fast5"
CLASSIFIER_DETAILS_FAST5_FILE = "../tests/data/classifier_details_test.fast5"
FILTER_FAST5_FILE = "../tests/data/filter_and_store_result_test.fast5"

raw_signal = np.random.randn(10)
raw = RawSignal(raw_signal, CHANNEL_NUMBER, CALIBRATION)
