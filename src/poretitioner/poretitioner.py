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
    LengthFilter,
    MaximumFilter,
    MeanFilter,
    MedianFilter,
    MinimumFilter,
    RangeFilter,
    StandardDeviationFilter,
)

    

def run(args):
    # Configures the root application logger.
    # After this line, it's safe to log using poretitioner.logger.getLogger() throughout the application.
    logger.configure_root_logger(verbosity=args.verbose, debug=args.debug)
    log = logger.getLogger()
    log.debug(f"Starting poretitioner with arguments: {vars(args)!s}")

    # Read configuration file, if it exists.
    try:
        configuration_path = Path(getattr(args, ARG.CONFIG)).resolve()
        print(f"\n\n\config path: {configuration_path!s}")
        configuration = readconfig(configuration_path)
        print(str(configuration))
    except AttributeError:
        log.info(f"No config file found from arg: {ARG.CONFIG}.")

    if args.command == COMMAND.SEGMENT:
        bulk_f5_filepath = Path(getattr(args, ARG.FILE)).resolve()
        

        # TODO: Update with actual segmentation config. https://github.com/uwmisl/poretitioner/issues/73
        filters: List[RangeFilter] = [
            # MeanFilter(),
            # StandardDeviationFilter(),
            # MedianFilter(),
            # MinimumFilter,
            # MaximumFilter(),
            LengthFilter(10, np.inf)
        ]

        # config = {
        #     "compute": {"n_workers": 30},
        #     "segment": {
        #         "voltage_threshold": -180,
        #         "signal_threshold_frac": 0.7,
        #         "translocation_delay": 10,
        #         "open_channel_prior_mean": 230,
        #         "open_channel_prior_stdv": 25,
        #         "good_channels": [1],
        #         "end_tol": 10,
        #         "terminal_capture_only": False,
        #     }}
        log.warning("")

        
        seg_config = configuration[CONFIG.SEGMENTATION]
        config = configuration[CONFIG.GENERAL]

        save_location = Path(getattr(args, ARG.OUTPUT_DIRECTORY)).resolve()
        
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
            f5_subsection_start=0,
            f5_subsection_end=None,
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
    use_fake_command_line = False
    if use_fake_command_line:
        command_line = [
            "segment",
            "--file=./src/tests/data/bulk_fast5_dummy.fast5",
            "--output-dir=./out/data/",
            "-vvvvv",
        ]
        args = get_args(command_line)
    else:
        args = get_args()
    # test_fast5()
    run(args)

if __name__ == "__main__":
    main()