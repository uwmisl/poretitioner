import argparse
from typing import Dict

import numpy as np
from poretitioner.utils.raw_signal_utils import compute_fractional_blockage

from . import logger


def main():
    args = get_args()

    # Configures the root application logger.
    logger.configure_root_logger(verbosity=args.verbose, include_debug_info=args.debug)

    log = logger.getLogger()
    log.debug("A debug message, mostly useful for developers")
    log.info("A general info message.")
    log.warning("a warning message, less verbose than just logging info")
    log.error("An error, this is always logged.")
    log.fatal("A fatal error - no recovery here. This is always logged.")
    x = np.array([1.0, 2.3, 4.0, 6.0, 10.0])
    frac = compute_fractional_blockage(x, 4)
    print(frac)


def get_args() -> Dict:
    """Gets the command line arguments passed to the application.

    Returns
    -------
    Dict
        Dictionary of command line arguments.
    """

    # TODO: Add a description string: https://github.com/uwmisl/poretitioner/issues/27
    DESCRIPTION = ""
    # TODO: Add a usage string: https://github.com/uwmisl/poretitioner/issues/27
    USAGE = ""

    # TODO: Add the rest of the commandline arguments and configuration: https://github.com/uwmisl/poretitioner/issues/27
    parser = argparse.ArgumentParser(description=DESCRIPTION, usage=USAGE)

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase the program's verbosity. Can be used multiple times to increase the logging level (e.g. -vvv for very verbose logging)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        default=False,
        help="Whether to run in debug mode. Turned off by default.",
    )

    # TODO: Add the rest of the commandline arguments and configuration: https://github.com/uwmisl/poretitioner/issues/27
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
