from math import pi, sin

import attrs

from rclinklab.base import FD, TimeService, TxSource
from rclinklab.converters import farray


@attrs.define
class SineSource(TxSource):
    frequency: float

    def __call__(self, time: int) -> FD:
        result = []
        for i in range(self.channels):
            phaseshift = i * 0.5 * pi
            result.append(sin((2 * pi * self.frequency * time / 1e6) + phaseshift))
        return farray(result)

    def start(self, time_service: TimeService) -> "TxSource":
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
