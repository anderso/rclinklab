from enum import Enum
from typing import Protocol

import psutil
import typer
from humanize import naturalsize
from rich import box
from rich.bar import Bar
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rclinklab.base import FD, Codec, Realtime, TxSource
from rclinklab.codecs.delta import DeltaCodec
from rclinklab.codecs.raw import RawCodec
from rclinklab.simulate import LinkPacket, PacketListener, RollingStatsCollector, Setup
from rclinklab.sources.functions import SineSource
from rclinklab.sources.joystick import JoystickTxSource
from rclinklab.stats import Stats

RATE_CHANNELS = 20
RATE_CODECS = 5
RATE_CPU = 2

app = typer.Typer()


class Source(str, Enum):
    joystick = "joystick"
    sine = "sine"


def _resolve_source(source):
    match source:
        case Source.joystick:
            return JoystickTxSource(channels=4)
        case Source.sine:
            return SineSource(frequency=0.5, channels=4)


@app.command()
def cli(source: Source):
    go(_resolve_source(source))


def go(source: TxSource):

    codecs = [
        RawCodec(channels=source.channels, bits=8),
        RawCodec(channels=source.channels, bits=9),
        RawCodec(channels=source.channels, bits=10),
        DeltaCodec(channels=source.channels, bits=10, delta_bits=5),
    ]
    setup = Setup(source=source, time_service=Realtime(), codecs=codecs)
    view = View(setup)
    live = Live(view.renderable, auto_refresh=False)

    listener = ViewPacketListener(view, live, rate_channels=RATE_CHANNELS, rate_codecs=RATE_CODECS, rate_cpu=RATE_CPU)
    setup.listeners = [listener]

    with live:
        setup.run()


class CliRepr(Protocol):
    def cli_repr(self) -> str:
        pass


class ChannelView:
    """Composes a text label, bar and numeric view of a channel."""

    def __init__(self, number):
        self.name = Text(f"CH{number}")
        self.bar = ChannelBar()
        self.data = Text()
        self.update(0)

    @property
    def renderables(self):
        return [self.name, self.bar, self.data]

    def update(self, value):
        self.bar.update(value)
        self.data.plain = f"{value:.3f}"


class ChannelBar(Bar):
    """Adapt a Bar to display values between -1.0 and 1.0."""

    def __init__(self):
        super().__init__(size=2, begin=1, end=1, width=50, color="deep_sky_blue4")

    def update(self, value):
        if value > 0:
            self.begin = 1
            self.end = value + 1
        else:
            self.begin = value + 1
            self.end = 1


class CodecRow:
    def __init__(self, codec: Codec):
        self.name = Text(repr(codec))
        self.mean_latency = Text()
        self.max_latency = Text()
        self.mean_error = Text()
        self.max_error = Text()

    @property
    def renderables(self):
        return [self.name, self.mean_latency, self.max_latency, self.mean_error, self.max_error]

    def update(self, stats: Stats):
        self.mean_latency.plain = f"{stats.latency.mean:.2f}"
        self.max_latency.plain = f"{stats.latency.max:.2f}"
        self.mean_error.plain = f"{stats.fd_error.mean:.6f}"
        self.max_error.plain = f"{stats.fd_error.max:.6f}"


class View:
    def __init__(self, setup: Setup):
        self.setup = setup
        self.channel_views = [ChannelView(i) for i in range(setup.source.channels)]
        self.bitrate = Text(str(setup.bitrate))
        self.packet_counter = Text()
        self.elapsed_time = Text()
        self.cpu_usage = Text()
        self.mem_usage = Text()
        self.codec_rows = [CodecRow(c) for c in self.setup.codecs]
        self.process = psutil.Process()

        grid = Table.grid()
        grid.add_row(self._channel_panel(), self._stats_panel())
        codec_table = self._codec_table(self.codec_rows)
        self.renderable = Group(grid, codec_table)

    def _channel_panel(self):
        grid = Table.grid()
        grid.add_column(width=3)
        grid.add_column()
        grid.add_column(width=6, justify="right")
        for cv in self.channel_views:
            grid.add_row(*cv.renderables)
        return Panel.fit(grid, title=repr(self.setup.source), title_align="left", box=box.SQUARE)

    def _stats_panel(self):
        grid = Table.grid()
        grid.add_column(width=12)
        grid.add_column(width=12, justify="right")
        grid.add_row("Bitrate", self.bitrate)
        grid.add_row("Elapsed time", self.elapsed_time)
        grid.add_row("Packet count", self.packet_counter)
        grid.add_row("CPU usage", self.cpu_usage)
        grid.add_row("Mem usage", self.mem_usage)
        return Panel.fit(grid, title="Stats", title_align="left", box=box.SQUARE)

    @staticmethod
    def _codec_table(rows):
        t = Table(box=box.SIMPLE_HEAD)
        t.add_column(header="Codec", width=60)
        t.add_column(header="Mean latency")
        t.add_column(header="Max latency")
        t.add_column(header="Mean error")
        t.add_column(header="Max error")
        for row in rows:
            t.add_row(*row.renderables)
        return t

    def update_stats(self, time, packets):
        seconds = time / 1_000_000
        self.elapsed_time.plain = f"{seconds:.3f}"
        self.packet_counter.plain = str(packets)

    def update_cpu(self):
        self.cpu_usage.plain = str(f"{self.process.cpu_percent():.0f} %")
        self.mem_usage.plain = naturalsize(self.process.memory_info().rss)

    def update_channels(self, data: FD):
        for cv, value in zip(self.channel_views, data):
            cv.update(value)

    def update_codec(self, codec_id, stats: Stats):
        self.codec_rows[codec_id].update(stats)


class ViewPacketListener(PacketListener):
    def __init__(self, view: View, live: Live, rate_channels, rate_codecs, rate_cpu):
        self.view = view
        self.live = live
        self.collector = RollingStatsCollector(time_limit=1_000_000)
        if not (rate_channels >= rate_codecs >= rate_cpu):
            raise ValueError()
        self.period_channels = self._rate_to_period(rate_channels)
        self.period_codecs = self._rate_to_period(rate_codecs)
        self.period_cpu = self._rate_to_period(rate_cpu)
        self.last_update_channels = 0
        self.last_update_codecs = 0
        self.last_update_cpu = 0
        self.packet_count = 0

    @staticmethod
    def _rate_to_period(rate):
        return round(1_000_000 / rate)

    def add(self, codec_id, packet: LinkPacket):
        current_time = packet.rx_ts
        self.collector.add(codec_id, packet)
        self.packet_count += 1
        if (current_time - self.last_update_channels) >= self.period_channels:
            self.view.update_channels(packet.tx_fd)
            self.view.update_stats(time=current_time, packets=self.packet_count)
            self.last_update_channels = current_time
            if (current_time - self.last_update_codecs) >= self.period_codecs:
                for codec_id, _ in enumerate(self.view.setup.codecs):
                    stats = self.collector.stats(codec_id)
                    self.view.update_codec(codec_id, stats)
                self.last_update_codecs = current_time
            if (current_time - self.last_update_cpu) >= self.period_cpu:
                self.view.update_cpu()
                self.last_update_cpu = current_time
            self.live.refresh()


if __name__ == "__main__":
    app()
