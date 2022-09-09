import itertools
import time
from collections import defaultdict, deque, namedtuple
from collections.abc import ValuesView
from functools import partial

import attrs
import evdev
import numpy as np
from evdev import AbsInfo, InputEvent, ecodes
from scipy.interpolate import interp1d

from rclinklab import base
from rclinklab.base import FD, LinkLabException, TxSource
from rclinklab.simulate import TimeService


@attrs.define
class Event:
    axis_id: int
    value: float
    ts: int


# TODO merge with the class above?
Pair = namedtuple("Pair", ["ts", "value"])


class InterpolatedEventStream:

    BUFFER_SIZE = 50

    def __init__(self):
        self.events: dict[int, list[Pair]] = defaultdict(partial(deque, maxlen=self.BUFFER_SIZE))

    def __call__(self, ts: int) -> FD:
        res = []
        for c in self.events.values():
            latest = c[-1]
            if ts >= latest.ts:
                res.append(latest.value)
            else:
                for p1, p2 in itertools.pairwise(reversed(c)):
                    if p2.ts <= ts < p1.ts:
                        res.append(interp1d(x=[p2.ts, p1.ts], y=[p2.value, p1.value])(ts))
                        break
                else:
                    raise ValueError(f"The requested timestamp {ts} is older than available events.")
        return np.array(res)

    def append(self, e: Event):
        self.events[e.axis_id].append(Pair(ts=e.ts, value=e.value))

    def latest(self):
        return max(c[-1].ts for c in self.events.values())


class JoystickTxSource(TxSource):

    start_ts: int
    device: evdev.InputDevice
    max_value: int
    axes_id_map: dict[int, int]

    def __init__(self, channels=None):
        """
        Args:
            channels: Optionally limit the number of axes, picks the first ones.
        """
        self.events = InterpolatedEventStream()
        super().__init__(channels)

    def __call__(self, ts: int) -> FD:
        if self.events.latest() < ts:
            self.read_events()
        return self.events(ts)

    def start(self, time_service: TimeService) -> "JoystickTxSource":
        self.start_ts = time_service.start_ts
        return self

    def __enter__(self) -> "JoystickTxSource":
        self.device, self.axes_id_map, self.max_value, initial_values = self.probe_device()
        if self.channels is None:
            self.channels = len(self.axes_id_map)
        else:
            self.axes_id_map = dict(list(self.axes_id_map.items())[: self.channels])
        for aid in self.axes_id_map:
            self.events.append(Event(axis_id=aid, value=self.map_value(initial_values[aid]), ts=0))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.device.close()

    def read_events(self):
        while event := self.device.read_one():
            match event:
                case InputEvent(type=ecodes.EV_ABS):  # type: ignore[misc]
                    self.events.append(self.map_input_event(event))
                case InputEvent(type=ecodes.EV_SYN, code=ecodes.SYN_DROPPED):  # type: ignore[misc]
                    # This indicates that we haven't read fast enough
                    raise LinkLabException("Input events where dropped from kernel buffer")

    @staticmethod
    def determine_resolution(max_value) -> int | None:
        """Assume min value = 0 and max = 255-511-1023 and so on"""
        for bits in range(4, 17):
            if max_value == 2**bits - 1:
                return bits
        raise LinkLabException(f"Could not determine resolution, {max_value=}")

    @staticmethod
    def check_assertions(axes: ValuesView[AbsInfo]):
        if {a.min for a in axes} != {0}:
            raise LinkLabException("Axes that do not start at 0 are not supported.")
        if len({a.max for a in axes}) != 1:
            raise LinkLabException("All axes must have the same range.")

    def map_value(self, value):
        return ((value / self.max_value) * 2) - 1

    def map_input_event(self, ie: InputEvent) -> Event:
        axis_id = self.axes_id_map[ie.code]
        value = self.map_value(ie.value)
        ts = (ie.sec * 1_000_000 + ie.usec) - self.start_ts
        return Event(axis_id=axis_id, value=value, ts=ts)

    def probe_device(self):
        """Find hid input device that support absolute axes."""
        base.log.info("Trying to connect to joystick...")
        while True:
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            for device in devices:
                match device.capabilities():
                    case {evdev.ecodes.EV_ABS: abs_list}:
                        abs_dict: dict[int, AbsInfo] = dict(abs_list)
                        self.check_assertions(abs_dict.values())
                        axes = list(abs_dict.values())
                        max_value = axes[0].max
                        resolution = self.determine_resolution(max_value)
                        base.log.info(f"Using {device.name} with {len(axes)} axes and {resolution} bit resolution")
                        axes_id_map = dict(enumerate(abs_dict.keys()))
                        initial_values = {aid: abs_dict[code].value for aid, code in axes_id_map.items()}
                        return device, axes_id_map, max_value, initial_values
            time.sleep(0.25)
