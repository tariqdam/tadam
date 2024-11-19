import datetime
from itertools import product
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


def assert_keys_in_object(obj: Any, keys: Iterable) -> None:
    """Check whether provided keys are in the object.

    Checks whether the provided elements in keys are present in the passed object and
    raises a KeyError with a list of missing elements.

    :param obj: Any object which can be processed as an iterable
    :param keys: items to check for their presence in obj
    """
    missing_keys: list = []
    for col in keys:
        if col not in obj:
            missing_keys.append(col)

    if len(missing_keys) > 0:
        raise KeyError(f"{missing_keys.__str__()} not found in object")
    return


def _check_type_compatibility(
    type_1: type, type_2: type, datetime_matches_timedelta: Optional[bool] = False
) -> bool:
    """Check if two types are compatible.

    If datetime_matches_timedelta is True, then
    datetime and timedelta are considered compatible.

    :param type_1: first type
    :param type_2: second type
    :param datetime_matches_timedelta: bool to compare datetime with timedelta
    :return: True if compatible, False otherwise
    """

    numerics = [
        int,
        float,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        pd.Float32Dtype(),
        pd.Float64Dtype(),
        pd.Int64Dtype(),
        pd.Int32Dtype(),
        pd.Int16Dtype(),
        pd.Int8Dtype(),
    ]
    date_times = [
        datetime.datetime,
        pd.Timestamp,
        pd.DatetimeTZDtype,
        np.dtypes.DateTime64DType,
        np.dtype("<M8[ns]"),
    ]
    time_deltas = [
        datetime.timedelta,
        pd.Timedelta,
        np.timedelta64,
        np.dtypes.TimeDelta64DType,
    ]

    if type_1 in numerics and type_2 in numerics:
        return True
    elif datetime_matches_timedelta:
        if type_1 in date_times and type_2 in time_deltas:
            return True
        elif type_1 in time_deltas and type_2 in date_times:
            return True
        else:
            return False
    elif type_1 in date_times and type_2 in date_times:
        return True
    elif type_1 in time_deltas and type_2 in time_deltas:
        return True
    else:
        return False


def assert_type_compatibility(
    data: pd.DataFrame,
    columns: list[str | int | float | tuple],
    values: Optional[list[str | int | float | pd.Timedelta]] = None,
) -> bool:
    """Check if all columns and values have compatible types.

    If columns contain datetime, then values can be timedelta and vice versa. To be used
     for comparing a dataframe where columns contain Timestamp data and values contain
     Timedelta data, formatted either as pd.Timestamp, np.datetime64 or numeric values
     for columns and pd.Timedelta, np.timedelta64 or numeric values for values.

    :param data: pandas DataFrame
    :param columns: list of columns to check dtypes for
    :param values: optional list of values to check types for
    :return: bool if all types are compatible else raise TypeError
    """
    assert_keys_in_object(obj=data.columns, keys=columns)
    assert len(columns) > 0, "columns must not be empty"

    column_dtypes = data[columns].dtypes.to_dict()

    incompatible_types = list()
    for (col_1, dtype_1), (col_2, dtype_2) in product(
        column_dtypes.items(), column_dtypes.items()
    ):
        if not _check_type_compatibility(
            type_1=dtype_1, type_2=dtype_2, datetime_matches_timedelta=False
        ):
            incompatible_types.append(((col_1, dtype_1), (col_2, dtype_2)))
    if values:
        for value_1, value_2 in product(values, values):
            if not _check_type_compatibility(
                type_1=type(value_1),
                type_2=type(value_2),
                datetime_matches_timedelta=False,
            ):
                incompatible_types.append(
                    ((value_1, type(value_1)), (value_2, type(value_2)))
                )
        for value, (col, dtype) in product(values, column_dtypes.items()):
            if not _check_type_compatibility(
                type_1=type(value), type_2=dtype, datetime_matches_timedelta=True
            ):
                incompatible_types.append(((value, type(value)), (col, dtype)))
    if incompatible_types:
        raise TypeError(f"Incompatible types: {incompatible_types.__repr__()}")
    else:
        return True


def assert_stop_after_start(
    data: pd.DataFrame,
    start: str | int | float | tuple,
    stop: str | int | float | tuple,
    allow_empty_values: bool | list[bool] | pd.Series = False,
) -> bool:
    """Check if all stop timestamps are after start timestamps

    Raise errors if not.

    :param data: pandas DataFrame with columns [start, stop]
    :param start: name of column containing start timestamps
    :param stop: name of column containing stop timestamps
    :param allow_empty_values: bool to suppress the assertion for empty start or stop
     timestamps. If a list of bools is passed with length two, the assertion is
     suppressed for [start, stop] columns. If a pd.Series of bools is passed equal to
     the length of the DataFrame, the assertion is suppressed for the corresponding rows
     in the dataframe.
    :return: True if all stop timestamps are after start timestamps
    """

    if isinstance(allow_empty_values, bool):
        allow_empty_values = [allow_empty_values, allow_empty_values]
    elif isinstance(allow_empty_values, pd.Series):
        assert len(allow_empty_values) == len(data)
    elif isinstance(allow_empty_values, list):
        assert len(allow_empty_values) == 2
    else:
        raise TypeError("allow_empty_values must be bool, list or pd.Series")

    assert_keys_in_object(obj=data.columns, keys=[start, stop])

    if isinstance(allow_empty_values, list):
        if not allow_empty_values[0]:
            assert data[start].notna().all(), f"{start} column contains NaN values"
        if not allow_empty_values[1]:
            assert data[stop].notna().all(), f"{stop} column contains NaN values"
    elif isinstance(allow_empty_values, pd.Series):
        # list
        filtered_data = data[~allow_empty_values]
        assert filtered_data[start].notna().all(), f"{start} column contains NaN values"
        assert filtered_data[stop].notna().all(), f"{stop} column contains NaN values"

    data_stop_after_start = (data[stop] - data[start]) < 0
    if data_stop_after_start.any():
        raise AssertionError("Not all stop timestamps are after start timestamps")
    else:
        return True
