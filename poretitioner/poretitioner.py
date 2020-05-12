from getargs import STEPS, get_args

from . import logger
from .utils import raw_signal_utils


def main():
    args = get_args()

    # Configures the root application logger.
    # After this line, it's safe to log using poretitioner.logger.getLogger() throughout the application.
    logger.configure_root_logger(verbosity=args.verbose, debug=args.debug)

    if args.command == STEPS.SEGMENT:
        # TODO: Perform segmentation step.
        pass
    elif args.command == STEPS.FILTER:
        # TODO: Perform filter step.
        pass
    elif args.command == STEPS.CLASSIFY:
        # TODO: Perform classification step.
        pass
    elif args.command == STEPS.QUANTIFY:
        # TODO: Perform quantification step.
        pass
    else:
        # TODO: Perform all steps
        pass


if __name__ == "__main__":
    main()
