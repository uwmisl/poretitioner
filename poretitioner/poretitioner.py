import numpy as np
from getargs import COMMAND, get_args
from signals import (
    BaseSignal,
    Capture,
    Channel,
    ChannelCalibration,
    FractionalizedSignal,
    PicoampereSignal,
    RawSignal,
    compute_fractional_blockage,
)

CALIBRATION = ChannelCalibration(0, 2, 1)
CHANNEL_NUMBER = 1


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
    signal = [1, 2, 3]
    raw = RawSignal(signal, CHANNEL_NUMBER, CALIBRATION)
    pico = raw.to_picoamperes()
    median = np.median(pico)

    raw = RawSignal(raw, CHANNEL_NUMBER, CALIBRATION)
    pico = PicoampereSignal(raw.to_picoamperes(), CHANNEL_NUMBER, CALIBRATION)
    open_channel_pA = np.median(pico)

    frac = FractionalizedSignal(pico, CHANNEL_NUMBER, CALIBRATION, open_channel_pA)
    converted_frac = pico.to_fractionalized(
        open_channel_guess=1, open_channel_bound=None, default=2
    )

    main()
