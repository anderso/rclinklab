import attrs
import numpy as np
import pandas as pd

from rclinklab.utils import extract_channel_data


@attrs.define
class BasicStats:

    max: float
    mean: float

    @classmethod
    def from_df(cls, data: pd.DataFrame):
        v: np.ndarray = data.values.reshape([-1])
        return cls(max=v.max(), mean=v.mean())


@attrs.define(init=False)
class Stats:
    total_packets: int
    packet_length_counts: pd.Series
    latency: BasicStats
    fd_error: BasicStats


def packets(rx_data, stats):
    stats.total_packets = rx_data.shape[0]
    stats.packet_length_counts = rx_data["ota_data", "ota_data"].apply(len).value_counts()


def latency(rx_data, stats):
    stats.latency = BasicStats.from_df((rx_data["rx_ts", "rx_ts"] - rx_data["tx_ts", "tx_ts"]))


def fd_error(rx_data, stats):
    differences = (extract_channel_data(rx_data, "rx_fd") - extract_channel_data(rx_data, "tx_fd")).abs()
    stats.fd_error = BasicStats.from_df(differences)


def calculate(data: pd.DataFrame) -> Stats:
    stats = Stats()
    packets(data, stats)
    latency(data, stats)
    fd_error(data, stats)
    return stats
