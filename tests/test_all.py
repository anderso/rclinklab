from pathlib import Path

from rclinklab.codecs.delta import DeltaCodec
from rclinklab.codecs.raw import RawCodec
from rclinklab.graph import graph
from rclinklab.simulate import Collector, Setup
from rclinklab.sources.blackbox import parse
from rclinklab.sources.functions import SineSource
from rclinklab.stats import BasicStats, calculate
from rclinklab.utils import attrs_to_data_frame

channels = 4

sources = [
    SineSource(frequency=10, channels=channels),
    parse(Path(__file__).parent / "blackbox-logs/short.bbl.csv"),
]

codecs = [
    DeltaCodec(channels=channels, bits=10, delta_bits=8),
    RawCodec(channels=channels, bits=10),
]


def test_all():
    """Excercise all sources and codecs against each other."""
    setup = Setup(codecs=codecs, duration=3_000_000)

    for source in sources:
        collector = Collector()
        setup.listeners = [collector]
        setup.source = source
        setup.run()
        for codec_id, p in collector.packets.items():
            df = attrs_to_data_frame(p)
            graph(source, df, show=False)
            stats = calculate(df)
            if isinstance(source, SineSource) and isinstance(setup.codecs[codec_id], RawCodec):
                assert stats.total_packets == 1500
                assert stats.latency == BasicStats(max=2000, mean=2000.0)
                assert stats.fd_error == BasicStats(max=0.000977517106549365, mean=0.0004986773515095856)
