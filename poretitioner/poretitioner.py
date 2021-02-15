from pathlib import Path

from . import logger
from .getargs import ARG, COMMAND, get_args
from .utils import segment

# from .fast5s import BulkFile, CaptureFile


# def test_fast5():

#     test_bulk_fas5_filepath = "tests/data/bulk_fast5_dummy.fast5"
#     test_capture_fast5_filepath = "tests/data/reads_fast5_dummy_9captures.fast5"
#     bulk = BulkFile(test_bulk_fas5_filepath)
#     calib = bulk.get_channel_calibration(1)
#     read_id = "read_8e8181d2-d749-4735-9cab-37648b463f88"
#     cap = CaptureFile(test_capture_fast5_filepath)
#     cap.get_capture_metadata_for_read(read_id)
#     print(calib)


def main(args):
    # Configures the root application logger.
    # After this line, it's safe to log using poretitioner.logger.getLogger() throughout the application.
    logger.configure_root_logger(verbosity=args.verbose, debug=args.debug)
    log = logger.getLogger()
    log.debug(f"Starting poretitioner with arguments: {args!s}")

    if args.command == COMMAND.SEGMENT:
        bulk_f5_filepath = Path(getattr(args, ARG.FILE)).resolve()
        save_location = Path(getattr(args, ARG.OUTPUT_DIRECTORY)).resolve()

        # TODO: Update with actual segmentation config. https://github.com/uwmisl/poretitioner/issues/73
        config = {
            "compute": {"n_workers": 4},
            "segment": {
                "voltage_threshold": -180,
                "signal_threshold": 0.7,
                "translocation_delay": 10,
                "open_channel_prior_mean": 230,
                "open_channel_prior_stdv": 25,
                "good_channels": [1, 2, 3],
                "end_tol": 0,
                "terminal_capture_only": False,
            },
            "filters": {"base filter": {"length": (100, None)}},
            "output": {"capture_f5_dir": f"{save_location}", "captures_per_f5": 1000},
        }
        segment.segment(
            bulk_f5_filepath,
            save_location,
            config,
            f5_subsection_start=None,
            f5_subsection_end=None,
        )
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


if __name__ == "__main__":
    # To test the application with pre-configured command line arguments,
    # set `use_fake_command_line` to True and/or modify the `command_line` list
    # with whatever arguments you'd like:
    use_fake_command_line = False
    if use_fake_command_line:
        command_line = [
            "segment",
            "--file=./tests/data/bulk_fast5_dummy.fast5",
            "--output-dir=./tests/data/",
            "-vvvvv",
        ]
        args = get_args(command_line)
    else:
        args = get_args()
    # test_fast5()
    main(args)
