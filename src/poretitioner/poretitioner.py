# from .fast5s import BulkFile, CaptureFile
import pprint
from pathlib import Path
from typing import List

import numpy as np

from . import logger
from .getargs import ARG, COMMAND, get_args
from .utils import segment
from .utils.configuration import CONFIG, readconfig, SegmentConfiguration, GeneralConfiguration
from .utils.filtering import (
    get_plugins,
    LengthFilter,
    MaximumFilter,
    MeanFilter,
    MedianFilter,
    MinimumFilter,
    RangeFilter,
    StandardDeviationFilter,
)

class Poretitoner:
    general_config: GeneralConfiguration
    segment_config: SegmentConfiguration

    def __init__(self, config):
        pass

    def segment(self):
        pass

    def filter(self):
        pass

    def classify(self):
        pass

    def quantify(self):
        pass


from poretitioner import Poretitoner

porty = Poretitoner("../path/tpo/config.toml")

porty.segment()


def run(args):
    # Configures the root application logger.
    # After these line, it's safe to log using src.poretitioner.logger.getLogger() throughout the application.
    logger.configure_root_logger(verbosity=args.verbose, debug=args.debug)
    log = logger.getLogger()
    log.debug(f"Starting poretitioner with arguments: {vars(args)!s}")

    # Get the command line args as a dictionary.
    command_line_args = vars(args)
    if "capture_directory" not in command_line_args and getattr(args, ARG.CAPTURE_DIRECTORY, False):
        command_line_args["capture_directory"] = command_line_args[ARG.CAPTURE_DIRECTORY]

    # Read configuration file, if it exists.
    try:
        configuration_path = Path(command_line_args[ARG.CONFIG]).resolve()
    except KeyError as e:
        log.info(f"No config file found from arg: {ARG.CONFIG}.")
        raise e
    configuration = readconfig(configuration_path, command_line_args=command_line_args, log=log)

    if args.command == COMMAND.SEGMENT:
        bulk_f5_filepath = Path(command_line_args[ARG.BULK_FAST5]).resolve()

        seg_config = configuration[CONFIG.SEGMENTATION]
        filter_config = configuration[CONFIG.FILTER]
        config = configuration[CONFIG.GENERAL]

        filters = get_plugins(filter_configs)

        save_location = Path(getattr(args, ARG.CAPTURE_DIRECTORY)).resolve()
        
        segmentation_config_str = pprint.pformat(seg_config.__dict__)
        general_config_str = pprint.pformat(config.__dict__)
        log.warning(f"bulk_f5_filepath: {bulk_f5_filepath}")
        log.warning(f"\n\nSave location: {save_location}")
        log.warning(f"\n\n Segmentation config: {segmentation_config_str}")

        log.warning(f"\n\n General config: {general_config_str}")

        capture_metadata = segment.segment(
            bulk_f5_filepath,
            config,
            seg_config,
            save_location=save_location,
            filters=filters,
            sub_run_start_seconds=0,
            sub_run_end_seconds=None,
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
            "--output-dir",
            "./out/data/",
            "--config",
            "./poretitioner_config.toml"
            " ",
            "-vvvvv",
        ]
        args = get_args(command_line)
    else:
        args = get_args()
    # test_fast5()
    run(args)

if __name__ == "__main__":
    main()