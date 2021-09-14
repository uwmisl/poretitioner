"""
========================
getargs.py
========================

This module is responsible for parsing the application's commandline arguments.

"""

import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import *  # I know people don't like import *, but I think it has benefits for types (doesn't impede people from being generous with typing)

import importlib.resources as resources

def as_cli_arg(property: str) -> str:
    """We'd like to match command line arguments to their
    corresponding python variables, but sadly python doesn't
    allow variable/field names with hyphens. As such,
    we convert the underscores to hyphens when using
    command line args.

    Parameters
    ----------
    property : Variable or field name with underscores.

    Returns
    -------
    str
        String with underscores replaced by dashes.
    """
    return property.replace("_", "-")


@dataclass(frozen=True)
class COMMAND:
    """Available Poretitioner commands. e.g. poretitioner segment"""

    # Only run the 'segment' step.
    SEGMENT = "segment"
    # Only run the 'filter' step.
    FILTER = "filter"
    # Only run the 'classify' steps.
    CLASSIFY = "classify"
    # Only run the 'quantify' steps.
    QUANTIFY = "quantify"
    # Run all the steps.
    ALL = None


@dataclass(frozen=True)
class ARG:
    """Available Poretitioner arguments. e.g. config, output, file"""

    class GENERAL:
        CONFIG = "config"
        DEBUG = "debug"
        BULK_FAST5 = "bulkfast5"
        VERBOSE = "verbose"

        # Segmenter
        CAPTURE_DIRECTORY = "capture_directory"  # Argument on the command line has a dash, but the attribute uses underscore.

    class SEGMENT:
        BULKFAST5 = "bulkfast5"
        N_CAPTURES_PER_FILE = "n_captures_per_file"
        VOLTAGE_THRESHOLD = "voltage_threshold"
        SIGNAL_THRESHOLD_FRAC = "signal_threshold_frac"
        TRANSLOCATION_DELAY = "translocation_delay"
        TERMINAL_CAPTURE_ONLY = "terminal_capture_only"
        END_TOLERANCE = "end_tolerance"
        GOOD_CHANNELS = "good_channels"
        OPEN_CHANNEL_PRIOR_MEAN = "open_channel_prior_mean"
        OPEN_CHANNEL_PRIOR_STDV = "open_channel_prior_stdv"

    # Filtration
    class FILTER:
        FILTER_SET_NAME = "filter_set_name"

        LENGTH_MAX = "filter_length_max"
        LENGTH_MIN = "filter_length_min"

        MEAN_MAX = "filter_mean_max"
        MEAN_MIN = "filter_mean_min"

        STDEV_MAX = "filter_stdv_max"
        STDEV_MIN = "filter_stdv_min"

        MEDIAN_MAX = "filter_median_max"
        MEDIAN_MIN = "filter_median_min"

        MEDIAN_MAX = "filter_median_max"
        MEDIAN_MIN = "filter_median_min"

        MAX = "filter_max"
        MIN = "filter_min"

        # Add new arguments here as well.
        ALL = [
            LENGTH_MAX,
            LENGTH_MIN,
            MEAN_MAX,
            MEAN_MIN,
            STDEV_MAX,
            STDEV_MIN,
            MEDIAN_MAX,
            MEDIAN_MIN,
            MEDIAN_MAX,
            MEDIAN_MIN,
            MAX,
            MIN,
        ]


def get_args(commandline_args: List = None) -> Namespace:
    parser = get_parser(commandline_args=commandline_args)
    args = parser.parse_args(commandline_args)
    return args


def get_help():
    return get_args(["--help"])


def get_parser(commandline_args: List = None) -> ArgumentParser:
    """Gets the command line arguments passed to the application.

    Parameters
    ----------
    args : List, optional
        Command line arguments list. If none are provided,
        the argument parser uses sys.argv[1:]. by default None

    Returns
    -------
    Namespace
        Namespace containing the command line arguments.
    """

    # TODO: Add a description string: https://github.com/uwmisl/poretitioner/issues/27
    DESCRIPTION = "Poretitioner is a powerful library and command line tool for parsing and interpreting nanopore data."
    # TODO: Add a usage string: https://github.com/uwmisl/poretitioner/issues/27
    USAGE = ""

    # TODO: Add the rest of the commandline arguments and configuration: https://github.com/uwmisl/poretitioner/issues/27
    parser = ArgumentParser(description=DESCRIPTION, usage=USAGE)

    def add_verbosity_option_to_parser(parser: ArgumentParser):
        """Adds the verbosity option to a parser. Each time this option is added, verbosity is incremented.
        The verbosity is set to 0 by default.

        Parameters
        ----------
        parser : ArgumentParser
            Parser to give a verbosity option, increase the verbosity by one.
        """
        parser.add_argument(
            "-v",
            f"--{ARG.GENERAL.VERBOSE}",
            action="count",
            default=0,
            help="Increase the program's verbosity. Can be used multiple times to increase the logging level (e.g. -vvv for very verbose logging). Should be added after the subcommand (if any).",
        )

    def add_debug_option_to_parser(parser: ArgumentParser):
        """Adds a debug option to a parser.

        Parameters
        ----------
        parser : ArgumentParser
            Parser to give a debug option. This option will be False by default.
        """
        parser.add_argument(
            "-d",
            f"--{ARG.GENERAL.DEBUG}",
            action="store_true",
            default=False,
            help="Whether to run in debug mode. Turned off by default. Should be added after the subcommand (if any) if desired.",
        )

    def add_input_to_parser(parser: ArgumentParser):
        """Adds an input option to a parser.

        Parameters
        ----------
        parser : ArgumentParser
            Parser to give a file input option.
        """
        parser.add_argument(
            "-f",
            f"--{ARG.GENERAL.BULK_FAST5}",
            action="store",
            help="The fast5 file to run poretitioner on.",
        )

    def add_configuration_option_to_parser(parser: ArgumentParser):
        """Adds a configuration option to a parser.

        Parameters
        ----------
        parser : ArgumentParser
            Parser to give an output file option
        """
        default_config = resources.Path("DEFAULT_PORETITIONER_CONFIG.toml").absolute().resolve()
        parser.add_argument(
            f"--{ARG.GENERAL.CONFIG}",
            action="store",
            default=default_config,
            help="Configuration file to configure Poretitioner.",
        )

    # Creates subparsers for each poretitioner command (e.g.`poretitioner segment`).
    # By default, if no command is provided, all steps will be run.
    subparser = parser.add_subparsers(dest="command")

    # Segmenter
    parser_segment = subparser.add_parser(COMMAND.SEGMENT, description="Segment captures")
    add_capture_directory_option_to_parser(parser_segment)

    parser_filter = subparser.add_parser(COMMAND.FILTER, description="Filter captures")
    parser_classify = subparser.add_parser(COMMAND.CLASSIFY, description="Classify captures")
    parser_quantify = subparser.add_parser(COMMAND.QUANTIFY, description="Quantify captures")

    parsers = [parser, parser_filter, parser_segment, parser_classify, parser_quantify]
    for subparser in parsers:
        add_input_to_parser(subparser)
        add_configuration_option_to_parser(subparser)
        add_debug_option_to_parser(subparser)
        add_verbosity_option_to_parser(subparser)

    return parser


# Segmenter


def add_capture_directory_option_to_parser(parser: ArgumentParser):
    """Adds output directory option to a parser.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to give an output directory option. This is where capture files will be saved.
    """
    arg = as_cli_arg(ARG.GENERAL.CAPTURE_DIRECTORY)
    parser.add_argument(
        f"--{arg}",
        action="store",
        default=os.environ.get("PWD"),
        help="Which directory to store the segmented capture fast5 files.",
    )


# Filters

# Mapping from command line option "--foo" to all its args and kwa
GENERAL_ARGS = {
    ARG.FILTER.FILTER_SET_NAME: {
        "action": "store",
        "help": "A unique identifier for a collection of filters, ideally describing why the collection was chosen, like 'NTER_PAPER_FINAL_2018_10_09'.",
        "type": str,
    },
    ARG.FILTER.LENGTH_MIN: {
        "action": "store",
        "help": "Exclude potential captures that have fewer than this many observations in the nanopore current trace.",
        "type": int,
    },
}


# Mapping from command line option "--foo" to all its args and kwa
FILTER_ARGS = {
    ARG.FILTER.FILTER_SET_NAME: {
        "action": "store",
        "help": "A unique identifier for a collection of filters, ideally describing why the collection was chosen, like 'NTER_PAPER_FINAL_2018_10_09'.",
        "type": str,
    },
    ARG.FILTER.LENGTH_MIN: {
        "action": "store",
        "help": "Exclude potential captures that have fewer than this many observations in the nanopore current trace.",
        "type": int,
    },
}

SEGMENTATION_ARGS = {
    ARG.SEGMENT.N_CAPTURES_PER_FILE: {
        "action": "store",
        # Katie Q: What units are the length in? Samples?
        "help": "How many captures to store in a file. When the number of captures founds exceeds this number (n), new files will be created to house the other n+1 through 2n, 2n+1 through 3n +1, ... and so on.",
        "type": int,
    },
    ARG.SEGMENT.OPEN_CHANNEL_PRIOR_MEAN: {
        "action": "store",
        # Katie Q: I don't feel like I'm phrasing this well, any ideas on better phrasing?
        "help": "Average open channel current in picoAmperes (e.g. 235).",
        "type": int,
    },
    ARG.SEGMENT.OPEN_CHANNEL_PRIOR_STDV: {
        "action": "store",
        # Katie Q: I don't feel like I'm phrasing this well, any ideas on better phrasing?
        "help": "Average open channel current in picoAmperes (e.g. 25).",
        "type": int,
    },
    ARG.SEGMENT.SIGNAL_THRESHOLD_FRAC: {},
    ARG.SEGMENT.TERMINAL_CAPTURE_ONLY: {},
    ARG.SEGMENT.TRANSLOCATION_DELAY: {},
    ARG.SEGMENT.VOLTAGE_THRESHOLD: {},
}
