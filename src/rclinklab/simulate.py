import math
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from statistics import mean

import attrs
from bitarray import bitarray

from rclinklab.converters import f2i_s, i2f_s

from . import base
from .base import FD, ID, Codec, SimulatedTime, TimeService, TxSource
from .stats import BasicStats, Stats

DEFAULT_BITRATE = 20_000  # bits per second


def bits_to_ts(bits, bitrate) -> int:
    return round((bits * 1_000_000) / bitrate)


@attrs.define
class TxData:
    codec_id: int
    start: int
    tx_ts: int
    tx_fd: FD
    tx_id: ID
    ota_data: bitarray


@attrs.define(init=False)
class LinkPacket:
    """Represents everything about a packet sent across the link, including the time"""

    tx_ts: int
    tx_fd: FD
    tx_id: ID
    ota_data: bitarray
    rx_id: ID
    rx_fd: FD
    rx_ts: int

    def __init__(self, tx_data: TxData, rx_ts, rx_id, rx_fd):
        self.tx_ts = tx_data.tx_ts
        self.tx_fd = tx_data.tx_fd
        self.tx_id = tx_data.tx_id
        self.ota_data = tx_data.ota_data
        self.rx_ts = rx_ts
        self.rx_id = rx_id
        self.rx_fd = rx_fd


class PacketListener(ABC):
    @abstractmethod
    def add(self, codec_id: int, packet: LinkPacket):
        pass


class Collector(PacketListener):
    def __init__(self, time_limit=None):
        self.packets: dict[int, deque[LinkPacket]] = defaultdict(deque)
        self.time_limit = time_limit

    def add(self, codec_id: int, packet: LinkPacket):
        p = self.packets[codec_id]
        p.append(packet)
        if self.time_limit is not None:
            while packet.rx_ts - p[0].rx_ts >= self.time_limit:
                del p[0]


@attrs.define
class PacketMetric:
    rx_ts: int
    latency: int
    max_error: float
    mean_error: float


class RollingStatsCollector(PacketListener):
    def __init__(self, time_limit):
        self.metrics: dict[int, list[PacketMetric]] = defaultdict(list)
        self.time_limit = time_limit

    def add(self, codec_id: int, p: LinkPacket):
        errors = abs(p.rx_fd - p.tx_fd)
        pm = PacketMetric(rx_ts=p.rx_ts, latency=(p.rx_ts - p.tx_ts), max_error=max(errors), mean_error=mean(errors))
        self.metrics[codec_id].append(pm)

    def stats(self, codec_id) -> Stats:
        codec_metrics = self.metrics[codec_id]
        newest = codec_metrics[-1].rx_ts
        for i, pm in enumerate(codec_metrics):
            if newest - pm.rx_ts <= self.time_limit:
                del codec_metrics[:i]
                break
        sum_latency = max_latency = 0
        sum_error = max_error = 0.0
        for pm in codec_metrics:
            sum_latency += pm.latency
            max_latency = max(max_latency, pm.latency)
            sum_error += pm.mean_error
            max_error = max(max_error, pm.max_error)
        stats = Stats()
        stats.latency = BasicStats(max_latency, sum_latency / len(codec_metrics))
        stats.fd_error = BasicStats(max_error, sum_error / len(codec_metrics))
        return stats


class Setup:
    def __init__(
        self,
        source: TxSource | None = None,
        codecs: list[Codec] | None = None,
        listeners: list[PacketListener] | None = None,
        bitrate: int = DEFAULT_BITRATE,
        duration: int | None = None,
        time_service: TimeService = SimulatedTime(),
    ):
        self.source: TxSource = source  # type: ignore
        self.codecs: list[Codec] = codecs  # type: ignore
        self.listeners: list[PacketListener] = listeners  # type: ignore
        self.bitrate: int = bitrate
        self.duration: int = duration  # type: ignore
        self.time_service: TimeService = time_service

    def run(self):
        Simulator.simulate(self)


class TransmitQueue:
    def __init__(self):
        self.queue = []

    def next(self) -> tuple[int, TxData]:
        return self.queue.pop(0)

    def transmit(self, data: TxData):
        self.queue.append((data.start + len(data.ota_data), data))
        self.queue.sort(key=lambda e: e[0])


class Simulator:
    @staticmethod
    def _transmit(start, source, codec_id, setup):
        codec = setup.codecs[codec_id]
        tx_ts = bits_to_ts(start, setup.bitrate)
        tx_fd = source(tx_ts)
        tx_id = f2i_s(tx_fd, codec.bits)
        ota_data = codec.transmit(tx_id)
        return TxData(codec_id, start, tx_ts, tx_fd, tx_id, ota_data)

    @staticmethod
    def _receive(tx_data: TxData, setup):
        codec = setup.codecs[tx_data.codec_id]
        rx_id = codec.receive(tx_data.ota_data)
        rx_fd = i2f_s(rx_id, codec.bits)
        return rx_id, rx_fd

    @staticmethod
    def _notify_listeners(codec_id, packet: LinkPacket, setup: Setup):
        for listener in setup.listeners:
            listener.add(codec_id, packet)

    @classmethod
    def simulate(cls, setup: Setup):
        base.log.info(f"Starting simulation using {repr(setup.source)}")

        duration_in_bits = setup.duration and math.ceil((setup.bitrate * setup.duration) / 1_000_000)
        position = 0  # track position in the bitstream

        queue = TransmitQueue()

        with setup.source.start(setup.time_service) as source:
            # Transmit for each codec at position = 0, seeding the queue
            for codec_id, _ in enumerate(setup.codecs):
                queue.transmit(cls._transmit(position, source, codec_id, setup))
            while True:
                position, tx_data = queue.next()
                rx_ts = bits_to_ts(position, setup.bitrate)
                setup.time_service.wait_until(rx_ts)
                rx_id, rx_fd = cls._receive(tx_data, setup)
                cls._notify_listeners(
                    tx_data.codec_id, LinkPacket(tx_data, rx_ts=rx_ts, rx_id=rx_id, rx_fd=rx_fd), setup
                )

                if duration_in_bits is not None and position >= duration_in_bits:
                    break

                queue.transmit(cls._transmit(position, source, tx_data.codec_id, setup))
