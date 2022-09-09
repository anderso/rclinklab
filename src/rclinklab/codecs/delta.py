import attrs
import numpy as np
from bitarray import bitarray

from rclinklab.base import ID, Codec
from rclinklab.converters import b2i_s, i2b_s, join, split


class State:
    def __init__(self, channels):
        self.last: ID = np.zeros(channels, dtype=np.int_)


def fit(value, bits):
    """If the value is outside the range of an unisgned int of size bits, adjust it to min/max."""
    return min(max(value, -(2 ** (bits - 1))), 2 ** (bits - 1) - 1)


@attrs.define(slots=False)
class DeltaCodec(Codec):

    delta_bits: int

    def __attrs_post_init__(self):
        self.tx_state = State(self.channels)
        self.rx_state = State(self.channels)

    def transmit(self, data: ID) -> bitarray:
        delta = data - self.tx_state.last
        delta = np.array([fit(v, self.delta_bits) for v in delta])
        self.tx_state.last += delta
        return join(i2b_s(delta, self.delta_bits, signed=True))

    def receive(self, data: bitarray) -> ID:
        delta = b2i_s(split(data, self.delta_bits), signed=True)
        new = self.rx_state.last + delta
        self.rx_state.last = new
        return new
