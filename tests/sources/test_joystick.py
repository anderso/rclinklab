from unittest.mock import Mock, sentinel

import numpy as np
from evdev import AbsInfo, InputDevice, InputEvent
from evdev.ecodes import EV_ABS
from pytest import approx

from rclinklab.simulate import SimulatedTime
from rclinklab.sources.joystick import Event, InterpolatedEventStream, JoystickTxSource

capabilities = {
    3: [
        (0, AbsInfo(value=0, min=0, max=2047, fuzz=7, flat=127, resolution=0)),
        (1, AbsInfo(value=2047, min=0, max=2047, fuzz=7, flat=127, resolution=0)),
    ],
}

input_events = [
    InputEvent(sec=0, usec=1000, type=EV_ABS, code=0, value=2047),
    InputEvent(sec=0, usec=1000, type=EV_ABS, code=1, value=0),
    InputEvent(sec=0, usec=2000, type=EV_ABS, code=0, value=1024 + 512),
    InputEvent(sec=0, usec=2000, type=EV_ABS, code=1, value=512),
    None,
]


def create_device_mock():
    device = Mock(InputDevice)
    device.capabilities = Mock(return_value=capabilities)
    device.name = "Mock joystick"
    device.read_one = Mock(side_effect=input_events)
    return device


def test_eventstream():
    s = InterpolatedEventStream()
    s.append(Event(axis_id=0, ts=0, value=2))
    assert s(0) == 2
    assert s(1) == 2
    s.append(Event(axis_id=0, ts=100, value=5))
    assert s(50) == 3.5


def fd_approx(values):
    return approx(np.array(values), rel=0.01)


def test_joystick(mocker):
    mocker.patch("evdev.list_devices", return_value=[sentinel.device_path])
    mocker.patch("evdev.InputDevice", return_value=create_device_mock())
    with JoystickTxSource().start(SimulatedTime()) as source:
        assert source(0) == fd_approx([-1.0, 1.0])
        assert source(500) == fd_approx([0.0, 0.0])
        assert source(1000) == fd_approx([1.0, -1.0])
        assert source(2000) == fd_approx([0.5, -0.5])
