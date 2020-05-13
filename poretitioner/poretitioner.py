from argparse import ArgumentParser, Namespace

from poretitioner.utils import streaming

from . import logger


def main():
    args = get_args()

    # Configures the root application logger.
    # After this line, it's safe to log using poretitioner.logger.getLogger() throughout the application.
    logger.configure_root_logger(verbosity=args.verbose, debug=args.debug)


def get_args() -> Namespace:
    """Gets the command line arguments passed to the application.

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
