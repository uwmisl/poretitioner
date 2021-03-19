"""
===================
playground.py
===================

Intended as a REPL environment to experiment small snippets of code, and testing how various app
components work together.

Best called with bpython [1] in interactive mode.
[1] - https://bpython-interpreter.org/
"""

import sys
from pathlib import Path
import importlib

project_root_dir = Path(__file__).parent.parent.resolve()
PROJECT_DIR_LOCATION = str(project_root_dir)


def add_poretitioner_to_path():
    poretitioner_directory = str(Path(PROJECT_DIR_LOCATION, "src"))
    sys.path.append(".")
    sys.path.append(poretitioner_directory)
    sys.path.append(PROJECT_DIR_LOCATION)


add_poretitioner_to_path()

import pprint
import numpy as np
import toml

from typing import *
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from os import PathLike

from src.poretitioner import BulkFile

from src.poretitioner.signals import *
from src.poretitioner.fast5s import *
from src.poretitioner.utils import *
from src.poretitioner.application_info import *
from src.poretitioner.getargs import *
import src.poretitioner.logger as logger
from src.poretitioner.logger import Logger as LoggerType, getLogger
from src.poretitioner.utils.classify import *
from src.poretitioner.utils.configuration import (
    GeneralConfiguration,
    SegmentConfiguration,
    readconfig,
)

from src import poretitioner
from src.poretitioner import CONFIG

LOG_VERBOSITY = 3
# Temporarily changing the logger format for the intro messages.
temp_logger_format = "%(log_color)s%(message)s"
logger.configure_root_logger(
    verbosity=LOG_VERBOSITY, debug=True, format=temp_logger_format
)

log = logger.getLogger()

THIS = __import__(__name__)  # The currently running module


log.debug(f"=============~ Poretitioner ~============\n")


def do_intro(log: LoggerType):
    log.debug(f"\nHi there, my name is Jessica!")
    log.info(f"And my name is Katie!")

    log.debug(f"\nWelcome to the Poretitioner Playground")
    log.debug(
        f"\nThis is a fun, interactive python REPL environment for learning about, testing, and hacking the Poretitioner."
    )
    log.debug(
        f"It's a great way to try new ideas, explore the Poretitioner APIs, or test your own code-- all without the hastle of building things from scratch."
    )

    log.debug(f"We've included lots of common modules like numpy--")
    log.info(f"And pytorch!")
    log.debug(f"So you can jump in, and build--")
    log.info(f"Or break!")
    log.debug(f"--things as fast as possible")

    log.debug(
        f"\nJust remember, if you make any changes to Poretitioner code, be sure to call: "
    )
    log.debug(f"")
    log.error(f"importlib.reload('poretitioner'")
    log.debug(f"")
    log.debug(f"So our runtime can pick up on your changes :)")
    log.debug(f"")


def explain_help(log: LoggerType):
    log.info("Need help? Not sure where to start?")
    log.info("Try typing 'help(poretitioner)'")
    log.info(
        "This works on any of the poretitioner APIs. e.g. help(poretitioner.segment)"
    )

    report_bugs(log)

    log.info(
        "\nFor more detailed help, feel free to email me at jdunstan@cs.washington.edu :)"
    )


class Helper:
    BASIC_HELP: str = """

    Need help? Not sure where to start?"

    Try typing 'help(poretitioner)'

    This works on any of the poretitioner APIs. e.g. help(poretitioner.segment)

    """

    def __init__(self, log: Optional[LoggerType] = None):
        self.log = log if log else getLogger()
        object.__setattr__(THIS, "help", self())
        object.__setattr__(THIS, "__help__", self())

        pass

    def basic_help(self):
        self.log.warn(self.BASIC_HELP)

    def __call__(self):
        self.basic_help()


class EmptyObj(object):
    pass


helper = EmptyObj()
setattr(helper, "__call__", explain_help)


def report_bugs(log: LoggerType):
    BUG_REPORT_LINK = (
        "https://github.com/uwmisl/poretitioner/issues/new?labels=beta-hackathon,beta-hackathon-bug"
    )
    FEEDBACK_LINK = "https://github.com/uwmisl/poretitioner/issues/new?labels=beta-hackathon,beta-hackathon-feedback"

    log.debug(f"Please report any bugs here: {BUG_REPORT_LINK}")
    log.debug(f"Please any general feedback here: {FEEDBACK_LINK}")


do_intro(log)
report_bugs(log)

# Quiet the logger while variables are being set up.
logger.configure_root_logger(verbosity=1, debug=False)

CALIBRATION = ChannelCalibration(0, 2, 1)
CHANNEL_NUMBER = 1
OPEN_CHANNEL_GUESS = 45
OPEN_CHANNEL_BOUND = 10
RANDOM_SIGNAL = np.random.random_integers(-200, high=800, size=30)
PORETITIONER_CONFIG_FILE = f"{PROJECT_DIR_LOCATION}/poretitioner_config.toml"
BULK_FAST5_FILE = f"{PROJECT_DIR_LOCATION}/src/tests/data/bulk_fast5_dummy.fast5"
CAPTURE_FAST5_FILE = (
    f"{PROJECT_DIR_LOCATION}/src/tests/data/reads_fast5_dummy_9captures.fast5"
)
CLASSIFIED_FAST5_FILE = (
    f"{PROJECT_DIR_LOCATION}/src/tests/data/classified_9captures.fast5"
)
CLASSIFIER_DETAILS_FAST5_FILE = (
    f"{PROJECT_DIR_LOCATION}/src/tests/data/classifier_details_test.fast5"
)
CLASSIFIED_10_MINS_4_CHANNELS = (
    f"{PROJECT_DIR_LOCATION}/src/tests/data/classified_10mins_4channels.fast5"
)
FILTER_FAST5_FILE = (
    f"{PROJECT_DIR_LOCATION}/src/tests/data/filter_and_store_result_test.fast5"
)


raw_signal = np.random.randn(10)
raw = RawSignal(raw_signal, CHANNEL_NUMBER, CALIBRATION)

bulky = BulkFile(BULK_FAST5_FILE, "r")

config = poretitioner.default_config(
    with_command_line_args={
        poretitioner.ARG.GENERAL.BULK_FAST5: "/Users/dna/Developer/poretitioner/src/tests/data/bulk_fast5_dummy.fast5",
        poretitioner.ARG.GENERAL.CAPTURE_DIRECTORY: "/tmp/captures",
    }
)

general_config = config[CONFIG.GENERAL]
segmentation_config = config[CONFIG.SEGMENTATION]
filter_config = config[CONFIG.FILTER]
classifier_config = config[CONFIG.CLASSIFICATION]


logger.configure_root_logger(verbosity=LOG_VERBOSITY, debug=True)


# capture_files = poretitioner.segment(general_config, segmentation_config)


cappy = CaptureFile(CLASSIFIED_FAST5_FILE, "a")

# classified = ClassifierFile(CLASSIFIED_10_MINS_4_CHANNELS)
# Do whatever else you'd like!
