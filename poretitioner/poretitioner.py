from . import logger
from .getargs import COMMAND, get_args
from .utils import raw_signal_utils


def main():
    args = get_args()

    # Configures the root application logger.
    # After this line, it's safe to log using poretitioner.logger.getLogger() throughout the application.
    logger.configure_root_logger(verbosity=args.verbose, debug=args.debug)
    log = logger.getLogger()
    log.debug(f"Starting poretitioner with arguments: {args!s}")

    if args.command == COMMAND.SEGMENT:
        # TODO: Perform segmentation step.
        pass
    elif args.command == COMMAND.FILTER:
        # TODO: Perform filter step.
        pass
    elif args.command == COMMAND.CLASSIFY:
        # TODO: Perform classification step.
        pass
    elif args.command == COMMAND.QUANTIFY:
        # TODO: Perform quantification step.
        pass
    else:
        # TODO: Perform all steps.
        pass


if __name__ == "__main__":
    main()
