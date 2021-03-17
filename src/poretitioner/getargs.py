"""
========================
getargs.py
========================

This module is responsible for parsing the application's commandline arguments.

"""
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import List


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

    CONFIG = "config"
    DEBUG = "debug"
    FILE = "file"
    BULK_FAST5 = "bulkfast5"
    OUTPUT = "output"
    VERBOSE = "verbose"

    # Segmenter
    CAPTURE_DIRECTORY = (
        "output_dir"  # Argument on the command line has a dash, but the attribute.
    )

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
    DESCRIPTION = ""
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
            f"--{ARG.VERBOSE}",
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
            f"--{ARG.DEBUG}",
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
            f"--{ARG.BULK_FAST5}",
            action="store",
            help="The fast5 file to run poretitioner on.",
        )

    def add_output_option_to_parser(parser: ArgumentParser):
        """Adds an output option to a parser.

        Parameters
        ----------
        parser : ArgumentParser
            Parser to give an output file option
        """
        parser.add_argument(
            "-o",
            f"--{ARG.OUTPUT}",
            action="store",
            help="Where to store this command's output.",
        )

    def add_configuration_option_to_parser(parser: ArgumentParser):
        """Adds a configuration option to a parser.

        Parameters
        ----------
        parser : ArgumentParser
            Parser to give an output file option
        """
        parser.add_argument(
            f"--{ARG.CONFIG}",
            action="store",
            default=".",
            help="Configuration file to configure Poretitioner.",
        )

    # Creates subparsers for each poretitioner command (e.g.`poretitioner segment`).
    # By default, if no command is provided, all steps will be run.
    subparser = parser.add_subparsers(dest="command")

    # Segmenter
    parser_segment = subparser.add_parser(
        COMMAND.SEGMENT, description="Segment captures"
    )
    add_capture_directory_option_to_parser(parser_segment)

    parser_filter = subparser.add_parser(COMMAND.FILTER, description="Filter captures")
    parser_classify = subparser.add_parser(
        COMMAND.CLASSIFY, description="Classify captures"
    )
    parser_quantify = subparser.add_parser(
        COMMAND.QUANTIFY, description="Quantify captures"
    )

    parsers = [parser, parser_filter, parser_segment, parser_classify, parser_quantify]
    for subparser in parsers:
        add_input_to_parser(subparser)
        add_configuration_option_to_parser(subparser)
        add_output_option_to_parser(subparser)
        add_debug_option_to_parser(subparser)
        add_verbosity_option_to_parser(subparser)

    args = parser.parse_args(commandline_args)
    return args


# Segmenter


def add_capture_directory_option_to_parser(parser: ArgumentParser):
    """Adds output directory option to a parser.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to give an output directory option. This is where capture files will be saved.
    """
    arg = as_cli_arg(ARG.CAPTURE_DIRECTORY)
    parser.add_argument(
        f"--{arg}",
        action="store",
        help="Which directory to store the segmented capture fast5 files.",
    )


# Filters
def add_capture_directory_option_to_parser(parser: ArgumentParser):
    """Adds output directory option to a parser.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to give an output directory option. This is where capture files will be saved.
    """
    arg = as_cli_arg(ARG.CAPTURE_DIRECTORY)
    parser.add_argument(
        f"--{arg}",
        action="store",
        help="Which directory to store the segmented capture fast5 files.",
    )


# Mapping from command line option "--foo" to all its args and kwa
FILTER_ARGS = {
    ARG.FILTER.LENGTH_MIN: {
        "action": "store",
        "help": "Exclude potential captures that have fewer than this many observations in the nanopore current trace.",
        "type": int,
    }
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
