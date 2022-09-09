import pytest

from rclinklab.codecs.delta import DeltaCodec
from rclinklab.converters import iarray

# TODO convert to generalized codec-tester


def _send_and_receive(data, codec):
    return codec.receive(codec.transmit(data))


def test_delta():
    codec = DeltaCodec(bits=10, channels=1, delta_bits=6)
    for value in [0, 10, 20, 0]:
        data = iarray([value])
        assert data == _send_and_receive(data, codec)


def test_delta_overflow_converges():
    codec = DeltaCodec(bits=8, channels=1, delta_bits=3)
    _send_and_receive(iarray([0]), codec)
    data = iarray([100])
    packets = 50
    for _ in range(packets):
        if data == _send_and_receive(data, codec):
            return
    pytest.fail(f"Did not converge within {packets} packets")
