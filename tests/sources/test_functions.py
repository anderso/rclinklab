from rclinklab.sources.functions import SineSource


def assert_channeldata(data):
    for value in data.tolist():
        assert -1.0 <= value <= 1.0


def test_sine():
    s = SineSource(frequency=1, channels=4)
    [assert_channeldata(s(ts)) for ts in range(0, 1_000_000, 100_000)]
