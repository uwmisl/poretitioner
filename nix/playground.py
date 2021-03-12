"""
===================
playground.py
===================

Makes it easy to test small snippets of code, testing how various app
components work together.
"""

import sys
import os
from pathlib import Path


project_root_dir = Path(__file__).parent.parent.resolve()
PROJECT_DIR_LOCATION = str(project_root_dir)


def add_poretitioner_to_path():
    print(f"\nProject directory: {PROJECT_DIR_LOCATION}\n")
    poretitioner_directory = str(Path(PROJECT_DIR_LOCATION, "src"))
    print(f"\nPoretitioner package location: {poretitioner_directory}\n\n")
    sys.path.append(".")
    sys.path.append(poretitioner_directory)
    sys.path.append(PROJECT_DIR_LOCATION)
    #os.chdir(str(Path(PROJECT_DIR_LOCATION, "src",)))


add_poretitioner_to_path()


import pprint
import numpy as np
import toml

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from os import PathLike
from typing import Any, Dict, List, Optional, Union

from src.poretitioner.signals import *
from src.poretitioner.fast5s import *
from src.poretitioner.utils import *
from src.poretitioner.application_info import *
from src.poretitioner.getargs import *
import src.poretitioner.logger as logger
from src.poretitioner.utils.classify import ClassifierFile
from src.poretitioner.utils.configuration import GeneralConfiguration, SegmentConfiguration, readconfig


LOG_VERBOSITY = 3
logger.configure_root_logger(verbosity=LOG_VERBOSITY, debug=True)

log = logger.getLogger()
log.debug(f"=============~ Poretitioner ~============")



CALIBRATION = ChannelCalibration(0, 2, 1)
CHANNEL_NUMBER = 1
OPEN_CHANNEL_GUESS = 45
OPEN_CHANNEL_BOUND = 10
RANDOM_SIGNAL = np.random.random_integers(-200, high=800, size=30)
PORETITIONER_CONFIG_FILE = f"{PROJECT_DIR_LOCATION}/poretitioner_config.toml"
BULK_FAST5_FILE = f"{PROJECT_DIR_LOCATION}/src/tests/data/bulk_fast5_dummy.fast5"
CAPTURE_FAST5_FILE = f"{PROJECT_DIR_LOCATION}/src/tests/data/reads_fast5_dummy_9captures.fast5"
CLASSIFIED_FAST5_FILE = f"{PROJECT_DIR_LOCATION}/src/tests/data/classified_9captures.fast5"
CLASSIFIER_DETAILS_FAST5_FILE = f"{PROJECT_DIR_LOCATION}/src/tests/data/classifier_details_test.fast5"
CLASSIFIED_10_MINS_4_CHANNELS = f"{PROJECT_DIR_LOCATION}/src/tests/data/classified_10mins_4channels.fast5"
FILTER_FAST5_FILE = f"{PROJECT_DIR_LOCATION}/src/tests/data/filter_and_store_result_test.fast5"

raw_signal = np.random.randn(10)
raw = RawSignal(raw_signal, CHANNEL_NUMBER, CALIBRATION)

classified = ClassifierFile(CLASSIFIED_10_MINS_4_CHANNELS)
# Do whatever else you'd like!