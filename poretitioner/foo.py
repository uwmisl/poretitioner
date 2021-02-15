import numpy as np
from signals import BaseSignal, Capture, FractionalizedSignal, PicoampereSignal, RawSignal
from utils.core import Channel, ChannelCalibration


class Boo:
    @property
    def foo(signal):
        return signal + 10


def main():
    calibration = ChannelCalibration(1, 1000, 100)
    channel = Channel(calibration, open_channel_guess=0, open_channel_bound=3)

    raw = [1, 2, 3]  # np.random.randn(10)
    frac = RawSignal(raw, channel)  # .to_picoamperes().to_fractionalized()

    print("okay")
    frac / 2.0
    frac + 1

    boo = Boo()
    boo.foo(10)


if __name__ == "__main__":
    main()
