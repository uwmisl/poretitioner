from typing import Sequence

from src.poretitioner.getargs import ARG, COMMAND, get_args

VERBOSE = ARG.VERBOSE
DEBUG = ARG.DEBUG
BULK_FAST5_FILE = ARG.BULK_FAST5


def args_from_str(string) -> Sequence[str]:
    """Helper method to convert a string to an arg sequence.

    Parameters
    ----------
    string : str
        String representing a command line argument.

    Returns
    -------
    Sequence[str]
        Sequence of strings parsed from the command string.
    """
    args = string.split(" ")
    return args


def get_args_no_subcommand_test():
    command = args_from_str(f"--{VERBOSE} --{DEBUG}")
    args = get_args(commandline_args=command)

    assert args.command == COMMAND.ALL, "When no subcommand is given, we should run all steps."
    assert args.debug, "Debug should be true when debug option is given."
    assert args.verbose == 1, "Verbose should be '1' when debug command is given."


def get_args_segment_subcommand_test():
    command = args_from_str(f"{COMMAND.SEGMENT} --{DEBUG}")
    args = get_args(commandline_args=command)

    assert (
        args.command == COMMAND.SEGMENT
    ), "When segment subcommand is given, we should run the segmentation steps."
    assert args.debug, "Debug should be true when debug option is given."


def get_args_verbose_3_test():
    command = args_from_str(f"{COMMAND.SEGMENT} --{VERBOSE} --{VERBOSE} --{VERBOSE}")
    args = get_args(commandline_args=command)

    assert (
        args.command == COMMAND.SEGMENT
    ), "When segment subcommand is given, we should run the segmentation steps."
    assert not args.debug, "Debug should be False when debug option is not given."
    assert args.verbose == 3, "Verbosity should be 3 when the option is added 3 times."


def get_args_file_test():
    test_filepath = "/User/foo/bar/rah.fast5"
    command = args_from_str(f"{COMMAND.QUANTIFY} --{VERBOSE} --{BULK_FAST5_FILE} {test_filepath}")
    args = get_args(commandline_args=command)

    assert (
        args.command == COMMAND.QUANTIFY
    ), "When quantify subcommand is given, we should run the quantify step."
    assert args.file == test_filepath, "Input file should be read when provided by the file option"
