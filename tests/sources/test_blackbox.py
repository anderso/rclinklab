from pathlib import Path

from pytest import approx

from rclinklab.sources.blackbox import parse

actual = {
    0: [-0.008, 0.004, -0.002, -1.000],
    905_867: [1.00000, -0.83800, 1.00000, -0.10000],
    2_310_574: [0.00200, -0.17800, -0.00200, 0.45400],
}

interpolated = {
    100_000: [0.10327, -0.08895, 0.10861, -0.90065],
    2_000_000: [0.22265, -0.32392, 0.21954, 0.33151],
}

no_data = {
    2_310_574 + 1: [0.0, 0.0, 0.0, 0.0],
    3_000_000: [0.0, 0.0, 0.0, 0.0],
}  # After the log has ended, the source should return 0


def test_blackbox():
    log_file = Path(__file__).parent / "../blackbox-logs/tiny.bbl.csv"
    source = parse(log_file)
    for ts, v in (actual | interpolated | no_data).items():
        assert source(ts) == approx(v, abs=1e-5)
