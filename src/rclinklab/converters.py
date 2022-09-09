"""
i = int (0 - 2^bits-1)
f = float (-1.0 - 1.0)
b = bitarray.bitarray (representing unsigned int)
a = np.ndarray
"""

import operator
from functools import reduce
from typing import Sequence

import numpy as np
from bitarray import bitarray
from bitarray.util import ba2int, int2ba

from rclinklab.base import FD, ID

# TODO remove _s-variants, merge with the single-value-variant, maybe using decorator?


def iarray(iterable) -> ID:
    return np.fromiter(iterable, dtype=int)


def farray(iterable) -> FD:
    return np.fromiter(iterable, dtype=float)


def b2a(value: bitarray, bits: int) -> ID:
    return iarray(b2i(v) for v in split(value, bits))


def f2i(value: float, bits: int) -> int:
    return round(((value + 1) / 2) * (2**bits - 1))


def i2f(value: int, bits: int) -> float:
    return ((value / (2**bits - 1)) * 2) - 1


def i2b(value: int, bits: int, signed=False) -> bitarray:
    return int2ba(value, bits, signed=signed)


def b2i(value: bitarray, signed=False) -> int:
    return ba2int(value, signed=signed)


def f2b(value: float, bits: int) -> bitarray:
    """
    >>> f2b(value=-1.0, bits=10)
    bitarray('0000000000')
    >>> f2b(value=0.0, bits=10)
    bitarray('1000000000')
    >>> f2b(value=1.0, bits=10)
    bitarray('1111111111')
    """
    return int2ba(f2i(value, bits), bits)


def b2f(value: bitarray) -> float:
    """
    >>> b2f(bitarray('0000'))
    -1.0
    >>> b2f(bitarray('1111'))
    1.0
    """
    return i2f(ba2int(value), len(value))


def i2f_s(values: ID, bits: int) -> FD:
    return farray(i2f(v, bits) for v in values.tolist())


def f2i_s(values: FD, bits: int) -> ID:
    return iarray(f2i(v, bits) for v in values)


def i2b_s(values: ID, bits: int, signed=False) -> Sequence[bitarray]:
    return [i2b(v, bits, signed=signed) for v in values.tolist()]


def b2i_s(values: Sequence[bitarray], signed=False) -> ID:
    return iarray(b2i(v, signed=signed) for v in values)


def join(values: Sequence[bitarray]) -> bitarray:
    return reduce(operator.add, values)


def split(ba: bitarray, bits: int) -> Sequence[bitarray]:
    result = []
    pieces = int(len(ba) / bits)
    for i in range(pieces):
        result.append(ba[i * bits : (i + 1) * bits])
    return result
