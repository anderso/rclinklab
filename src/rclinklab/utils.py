import attrs
import numpy as np
import pandas as pd


def extract_channel_data(df, name) -> pd.DataFrame:
    cdf = df[name].copy()
    cdf.columns = range(cdf.shape[1])
    return cdf


def attrs_to_data_frame(values):
    """Somewhat hacky way to create a multilevel dataframe from attrs introspection, flattening sequences.

    >>> @attrs.define
    ... class Data:
    ...     name: str
    ...     stuff: tuple[float, ...]
    ...
    >>> attrs_to_data_frame([Data(name="One", stuff=(1.2, 3.7, 2.8)), Data(name="Two", stuff=(3.3, 4.4, 5.5))])
      name    stuff
      name stuff[0] stuff[1] stuff[2]
    0  One      1.2      3.7      2.8
    1  Two      3.3      4.4      5.5
    """

    def attrs_to_multiindex(inst):
        result = []
        for name, value in attrs.asdict(inst).items():
            if isinstance(value, (tuple, list, np.ndarray)):
                result.extend((name, f"{name}[{i}]") for i in range(len(value)))
            else:
                result.append((name, name))
        return pd.MultiIndex.from_tuples(result)

    def flatten(inst):
        result = []
        for _, value in attrs.asdict(inst).items():
            if isinstance(value, (tuple, list, np.ndarray)):
                result.extend(value)
            else:
                result.append(value)
        return result

    index = attrs_to_multiindex(values[0])
    return pd.DataFrame.from_records((flatten(v) for v in values), columns=index)
