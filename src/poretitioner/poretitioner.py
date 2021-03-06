# from .fast5s import BulkFile, CaptureFile
import pprint
from pathlib import Path
from typing import *  # I know people don't like import *, but I think it has benefits for types (doesn't impede people from being generous with typing)

import logger
import numpy as np

from .getargs import ARG, COMMAND, get_args
from .utils import segment
from .utils.configuration import CONFIG, GeneralConfiguration, SegmentConfiguration, readconfig
from .utils.filtering import FilterConfig, FilterConfigs, Filters, FilterSet, get_filters


def run(args):
    # Configures the root application logger.
    # After these line, it's safe to log using src.poretitioner.logger.getLogger() throughout the application.
    logger.configure_root_logger(verbosity=args.verbose, debug=args.debug)
    log = logger.getLogger()
    log.debug(f"Starting poretitioner with arguments: {vars(args)!s}")

    # Get the command line args as a dictionary.
    command_line_args = vars(args)
    if "capture_directory" not in command_line_args and getattr(
        args, ARG.GENERAL.CAPTURE_DIRECTORY, False
    ):
        command_line_args["capture_directory"] = command_line_args[ARG.GENERAL.CAPTURE_DIRECTORY]

    # Read configuration file, if it exists.
    try:
        configuration_path = Path(command_line_args[ARG.GENERAL.CONFIG]).resolve()
    except KeyError as e:
        log.info(f"No config file found from arg: {ARG.GENERAL.CONFIG}.")
        raise e
    configuration = readconfig(configuration_path, command_line_args=command_line_args, log=log)
    bulk_f5_filepath = Path(command_line_args[ARG.GENERAL.BULK_FAST5]).resolve()

    seg_config = configuration[CONFIG.SEGMENTATION]
    filter_set: FilterSet = configuration[CONFIG.FILTER]
    config = configuration[CONFIG.GENERAL]

    save_location = Path(getattr(args, ARG.GENERAL.CAPTURE_DIRECTORY)).resolve()

    log.info(f"bulk_f5_filepath: {bulk_f5_filepath}")
    log.info(f"Save location: {save_location}")

    if args.command == COMMAND.SEGMENT:
        segmentation_config_str = pprint.pformat(seg_config.__dict__)
        general_config_str = pprint.pformat(config.__dict__)

        capture_metadata = segment.segment(
            bulk_f5_filepath,
            config,
            seg_config,
            save_location=save_location,
            sub_run_start_observations=0,
            sub_run_end_observations=None,
        )

        segment_results = pprint.pformat(capture_metadata)
        log.debug(f"All done segmenting: \n{segment_results}")
        return
    elif args.command == COMMAND.FILTER:
        # TODO: Perform filter step.
        return
    elif args.command == COMMAND.CLASSIFY:
        # TODO: Perform classification step.
        return
    elif args.command == COMMAND.QUANTIFY:
        # TODO: Perform quantification step.
        return
    else:
        # TODO: Perform all steps.
        return


def main():
    # To test the application with pre-configured command line arguments,
    # set `use_fake_command_line` to True and/or modify the `command_line` list
    # with whatever arguments you'd like:
    use_fake_command_line = True
    if use_fake_command_line:
        command_line = [
            "segment",
            "--bulkfast5",
            "./src/tests/data/bulk_fast5_dummy.fast5",
            "--capture-directory",
            "./out/data/",
            "--config",
            "./poretitioner_config.toml" " ",
            "-vvvvv",
        ]
        args = get_args(command_line)
    else:
        args = get_args()
    # test_fast5()
    run(args)
