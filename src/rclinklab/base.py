import time
from abc import ABC, abstractmethod

import attrs
import numpy as np
import numpy.typing as npt
import pandas as pd
import structlog
from bitarray import bitarray
from scipy.interpolate import interp1d

# TODO choose better names
FD = npt.NDArray[np.float_]  # floats with range -1.0 - 1.0
ID = npt.NDArray[np.int_]  # ints with range 0 - 2^bits-1

log = structlog.get_logger()


class TimeService(ABC):
    """Defines a start time and a way to wait for a timestamp.

    This is the mechanism that allows the simulator to operate in either
    realtime or simulation time, and provides testability.
    """

    start_ts: int

    @abstractmethod
    def wait_until(self, ts: int):
        pass


class Realtime(TimeService):
    def __init__(self):
        self.start_ts = self._time_us()

    @staticmethod
    def _time_us() -> int:
        return round(time.time_ns() / 1000)

    def wait_until(self, ts: int):
        # Make sure we are 50 ms after realtime to allow for hid events to arrive
        diff = ts - (self._time_us() - self.start_ts - 50_000)
        if diff > 0:
            time.sleep(diff / 1_000_000)


class SimulatedTime(TimeService):
    start_ts = 0

    def wait_until(self, ts: int):
        pass


@attrs.define
class TxSource(ABC):
    """Calling this with a timestamp should produce axis data."""

    channels: int

    @abstractmethod
    def __call__(self, time: int) -> FD:
        pass

    @abstractmethod
    def start(self, time_service: TimeService) -> "TxSource":
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def multi_index(self):
        return pd.MultiIndex.from_tuples(
            [("tx_ts", "tx_ts"), *[("tx_fd", f"tx_fd[{i}]") for i in range(self.channels)]]
        )

    def series(self, ts):
        return pd.Series([ts] + list(self(ts)))

    def data_frame(self, ts: pd.Series) -> pd.DataFrame:
        df = ts.apply(self.series)
        df.columns = self.multi_index()
        return df


class InterpolatedTxSource(TxSource):
    def __init__(self, data: pd.DataFrame):
        super().__init__(channels=len(data.columns) - 1)
        data.columns = self.multi_index()
        self._data = data
        self.interpolator = interp1d(x=data["tx_ts", "tx_ts"], y=data["tx_fd"], kind="linear", axis=0)

    def start(self, time_service: TimeService) -> "TxSource":
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, time: int) -> FD:
        if time > self.interpolator.x[-1]:
            return np.array([0.0 for _ in range(self.channels)])
        return self.interpolator(time)

    def raw_data(self, duration: int) -> pd.DataFrame:
        """Create a dataframe with the raw data up to a specific duration"""
        # TODO add zeros or loop if duration is larger than data
        d = self._data
        return d[d["tx_ts", "tx_ts"] <= duration]


@attrs.define
class Codec(ABC):

    channels: int
    bits: int

    @abstractmethod
    def transmit(self, data: np.ndarray) -> bitarray:
        pass

    @abstractmethod
    def receive(self, data: bitarray) -> np.ndarray:
        pass


class LinkLabException(Exception):
    """Just to gather exceptions explicitly thrown in this package"""

    pass
