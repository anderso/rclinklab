import pandas as pd
import plotly.graph_objects as go

from .base import InterpolatedTxSource, TxSource

config = {"scrollZoom": "cartesian"}


# TODO graph more than one channel


def source_scatter(source: TxSource, duration, step=1_000):
    if isinstance(source, InterpolatedTxSource):
        data = source.raw_data(duration)
    else:
        data = source.data_frame(pd.Series(range(0, duration, step)))
    return go.Scatter(x=data["tx_ts", "tx_ts"], y=data["tx_fd", "tx_fd[0]"], name="source", line_shape="linear")


def graph(source: TxSource, data: pd.DataFrame, show: bool = True):
    duration = data["tx_ts", "tx_ts"].iloc[-1]

    fig = go.Figure()
    fig.add_trace(source_scatter(source, duration))
    fig.add_trace(go.Scatter(x=data["rx_ts", "rx_ts"], y=data["rx_fd", "rx_fd[0]"], name="received", line_shape="hv"))
    fig.update_xaxes(rangeslider={"visible": True})
    if show:
        fig.show(renderer="browser", config=config)
