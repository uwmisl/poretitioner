"""
========================
getargs.py
========================

This module is responsible for parsing the application's commandline arguments.

"""
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class COMMAND:
    """Available Poretitioner commands. e.g. poretitioner segment
    """

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
    """Available Poretitioner arguments. e.g. config, output, file
    """

    CONFIG = "config"
    DEBUG = "debug"
    FILE = "file"
    OUTPUT = "output"
    VERBOSE = "verbose"


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
            "-f", f"--{ARG.FILE}", action="store", help="The fast5 file to run poretitioner on."
        )

    def add_output_option_to_parser(parser: ArgumentParser):
        """Adds an output option to a parser.

        Parameters
        ----------
        parser : ArgumentParser
            Parser to give an output file option
        """
        parser.add_argument(
            "-o", f"--{ARG.OUTPUT}", action="store", help="Where to store this command's output."
        )

    def add_configuration_option_to_parser(parser: ArgumentParser):
        """Adds a configuration option to a parser.

        Parameters
        ----------
        parser : ArgumentParser
            Parser to give an output file option
        """
        parser.add_argument(
            f"--{ARG.CONFIG}", action="store", help="Configuration file to configure Poretitioner."
        )

    # Creates subparsers for each poretitioner command (e.g.`poretitioner segment`).
    # By default, if no command is provided, all steps will be run.
    subparser = parser.add_subparsers(dest="command")
    parser_segment = subparser.add_parser(COMMAND.SEGMENT, description="Segment captures")
    parser_filter = subparser.add_parser(COMMAND.FILTER, description="Filter captures")
    parser_classify = subparser.add_parser(COMMAND.CLASSIFY, description="Classify captures")
    parser_quantify = subparser.add_parser(COMMAND.QUANTIFY, description="Quantify captures")

    parsers = [parser, parser_filter, parser_segment, parser_classify, parser_quantify]
    for subparser in parsers:
        add_input_to_parser(subparser)
        add_configuration_option_to_parser(subparser)
        add_output_option_to_parser(subparser)
        add_debug_option_to_parser(subparser)
        add_verbosity_option_to_parser(subparser)

    args = parser.parse_args(commandline_args)
    return args
