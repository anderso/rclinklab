"""Parse a blackbox log file exported to csv by Betaflight Blackbox Explorer
(https://github.com/betaflight/blackbox-log-viewer).

Optimally disable RC smoothing and use the same logging rate as the rc link
rate. So for example for 500Hz ELRS and 8kHz PID loop, use 1/16 logging rate.
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from rclinklab.base import InterpolatedTxSource, LinkLabException


def find_header_lineno(log_path) -> int:
    with open(log_path) as bb_log:
        for line_number, values in enumerate(csv.reader(bb_log)):
            if len(values) != 2:
                return line_number
    raise LinkLabException("Unexpected file format")


def read_csv(path):
    header_line_no = find_header_lineno(path)
    columns = ["time", "rcCommand[0]", "rcCommand[1]", "rcCommand[2]", "rcCommand[3]"]
    with open(path) as bb_log:
        return pd.read_csv(bb_log, header=header_line_no, usecols=columns, dtype=np.float64)


def adapt(data: pd.DataFrame):
    data["time"] -= data["time"][0]  # make timestamps start at 0
    data["rcCommand[3]"] -= 1500  # Adjust throttle interval
    data.iloc[:, 1:] /= 500  # Adjust range to -1.0 - 1.0
    return data


def parse(path: Path) -> InterpolatedTxSource:
    return InterpolatedTxSource(adapt(read_csv(path)))
