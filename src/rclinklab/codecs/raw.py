from bitarray import bitarray

from rclinklab.converters import b2i_s, i2b_s, join, split

from ..base import ID, Codec


class RawCodec(Codec):
    def transmit(self, data: ID) -> bitarray:
        return join(i2b_s(data, self.bits))

    def receive(self, data: bitarray) -> ID:
        return b2i_s(split(data, self.bits))
