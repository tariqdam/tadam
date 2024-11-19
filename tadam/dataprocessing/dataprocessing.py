"""
Functions in this file are intended to manipulate health record data.


Desired functions:

- transform single measurement records to time series records
-
"""

import warnings
from numbers import Real
from typing import Any, Callable, Optional, TypeVar

import duckdb as dd
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_datetime64_ns_dtype
from scipy.stats import linregress

from tadam.assertions import (
    assert_keys_in_object,
    assert_stop_after_start,
    assert_type_compatibility,
)

IFT = TypeVar("IFT", Real, pd.Timedelta)


def transform_single_to_range(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Transform single measurements to range measurements"""
    warnings.warn(
        "The function transform_single_to_range is deprecated. Use "
        "single_to_range instead.",
        DeprecationWarning,
    )
    return single_to_range(*args, **kwargs)


def transform_overlapping_to_discrete_series(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Transform overlapping series to discrete series"""
    warnings.warn(
        "The function transform_overlapping_to_discrete_series is deprecated."
        " Use overlap_to_discrete instead.",
        DeprecationWarning,
    )
    return overlap_to_discrete(*args, **kwargs)


def join_adjoining_records(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Transform boundary records with constant values into single records"""
    warnings.warn(
        "The function join_adjoining_records is deprecated."
        " Use join_adjoining instead.",
        DeprecationWarning,
    )
    return join_adjoining(*args, **kwargs)


def impute_missing_as_zero(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Create records with value zero for missing time intervals"""
    warnings.warn(
        "The function impute_missing_as_zero is deprecated."
        " Use impute_missing instead.",
        DeprecationWarning,
    )
    return impute_missing(*args, **kwargs)


def merge_windows_with_measurements(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Aggregate ranged measurements within the defined windows"""
    warnings.warn(
        "The function merge_windows_with_measurements is deprecated."
        " Use merge_windows instead.",
        DeprecationWarning,
    )
    return merge_windows(*args, **kwargs)


def impute_missing(
    data: pd.DataFrame,
    group_by: list[str],
    data_start: str,
    data_end: str,
    data_var: str,
    data_val: str,
    windows: pd.DataFrame | None = None,
    merge_on: list[str] | None = None,
    windows_start: str | None = None,
    windows_end: str | None = None,
    impute_window: str = "combined",
    discrete_method: str = "sum",
) -> pd.DataFrame:
    """
    Creates a dataframe with missing values imputed as zero. The dataframe
    contains the same columns as the data dataframe, but with the windows
    period covered. Any records in the data dataframe that
    :param data: pd.DataFrame with records of single measurements
    :param group_by: passed directly to pandas.groupby (see pandas documentation)
    :param data_start: column name of start timestamp
    :param data_end: column name of end timestamp
    :param data_var: column name of variable id
    :param data_val: column name of value
    :param windows: pd.DataFrame with windows to impute over (optional)
        This can be used for merging with admission periods or any specific windows.
    :param merge_on: column name to merge windows with data on.
        This merge is necessary to retrieve the correct group_by columns with each
        window record.
    :param windows_start: column name of start timestamp
    :param windows_end: column name of end timestamp
    :param impute_window: one of ['data', 'windows', 'combined']
        data: impute missing values within the data dataframe
        windows: impute missing values within the windows dataframe
        combined: impute missing values within the data dataframe and the windows
    :param discrete_method: passed to overlap_to_discrete
        sum: sum overlapping values, resulting in 0 if no values are present and in
            the original value otherwise.
    :return: pd.DataFrame with imputed values as well as original values
        columns: group_by + [data_start, data_end, data_var, data_val]
    """

    if isinstance(group_by, str):
        group_by = [group_by]

    IMPUTE_WINDOWS = ["data", "windows", "combined"]
    IMPUTE_ERROR = f"impute_window must be one of {IMPUTE_WINDOWS}, not {impute_window}"
    if impute_window not in IMPUTE_WINDOWS:
        raise ValueError(IMPUTE_ERROR)

    def _get_data() -> tuple[pd.DataFrame, pd.Series]:
        data_columns = group_by + [data_start, data_end, data_var, data_val]
        assert_keys_in_object(obj=data.columns, keys=data_columns)
        _d = data[data_columns]
        _vars = pd.Series(_d[data_var].unique(), name=data_var)
        return _d, _vars

    def _process_data() -> pd.DataFrame:
        return (
            _d.groupby(by=group_by)
            .agg({data_start: "min", data_end: "max"})
            .reset_index()
        )

    def _process_windows() -> pd.DataFrame:
        if windows is not None:
            return windows[windows_columns].rename(
                columns={
                    windows_start: data_start,
                    windows_end: data_end,
                }
            )
        else:
            raise ValueError("windows cannot be None")

    _d, _vars = _get_data()

    if impute_window == "data":
        if windows is not None:
            warnings.warn(
                f"With impute_window == {impute_window}, passed windows are ignored"
            )
        _i = _process_data()
    elif impute_window in ["windows", "combined"]:
        essential_kwargs = {
            "windows": windows,
            "merge_on": merge_on,
            "windows_start": windows_start,
            "windows_end": windows_end,
        }
        missing_kwargs = []
        for k, v in essential_kwargs.items():
            if v is None:
                missing_kwargs.append(k)
        if missing_kwargs:
            raise ValueError(
                f"With impute_window == {impute_window}, the following kwargs are "
                f"missing: {missing_kwargs}"
            )

        if isinstance(merge_on, str):
            merge_on = [merge_on]
        windows_columns = [x for x in [windows_start, windows_end] if x is not None]
        if merge_on:
            windows_columns = merge_on + windows_columns
        windows_columns = [x for x in windows_columns if x is not None]

        if windows is not None:
            assert_keys_in_object(obj=windows.columns, keys=windows_columns)

        _wi = _process_windows()
        _wi_merged = pd.merge(
            left=_wi,
            right=_d[group_by],
            on=merge_on,
            how="left",
            suffixes=("", "_y"),
        )[
            group_by + [data_start, data_end]
        ]  # PM: process_windows already renamed

        if impute_window == "windows":
            _i = _wi_merged
        elif impute_window == "combined":
            _dwi = pd.concat([_d, _wi], ignore_index=True, axis=0, copy=False)
            _i = (
                _dwi.groupby(group_by)
                .agg({data_start: "min", data_end: "max"})
                .reset_index()
            )
        else:
            raise ValueError(IMPUTE_ERROR)
    else:
        raise ValueError(IMPUTE_ERROR)

    _imputed = pd.merge(_i, _vars, how="cross")
    _imputed[data_val] = 0

    _concat = pd.concat(
        objs=[_d, _imputed],
        axis=0,
        ignore_index=True,
        copy=False,
    )
    _data = _concat.sort_values(
        by=group_by + [data_start, data_end, data_var, data_val],
    )
    _data = overlap_to_discrete(
        data=_data,
        timestamp_start=data_start,
        timestamp_end=data_end,
        value=data_val,
        group_by=group_by + [data_var],
        keep=discrete_method,
    )
    return _data


def single_to_range(
    data: pd.DataFrame,
    timestamp: Optional[str | int | float | tuple] = "timestamp",
    group_by: Optional[list[str | int | float | tuple]] = None,
    direction: Optional[str] = "forward",
    max_duration: Optional[IFT] = None,
    fill_duration: Optional[IFT] = None,
) -> pd.DataFrame:
    """
    :param data: pandas DataFrame with records of single measurements
    :param timestamp: column name of timestamp
    :param group_by: passed directly to pandas.groupby (see pandas documentation)
    :param direction: fill direction: 'forward' | 'backward'
    :param max_duration: if not None, maximum duration of each record in the output
    :param fill_duration: if not None, if a record is missing, fill with this duration
    :return:
    """

    passed_columns: list = [timestamp]

    if group_by:
        if None in group_by:
            raise ValueError("group_by cannot contain None")
        passed_columns += group_by
    assert_keys_in_object(obj=data.columns, keys=passed_columns)

    if direction not in ["forward", "backward"]:
        raise ValueError("direction must be either 'forward' or 'backward'")

    if max_duration and fill_duration:
        if type(max_duration) != type(fill_duration):
            raise TypeError(
                "If both max_duration and fill_duration are set, their types "
                "must match with eachother as well as the timestamp column"
            )
    if max_duration:
        if pd.api.types.is_numeric_dtype(max_duration):
            if not pd.api.types.is_numeric_dtype(data[timestamp]):
                raise TypeError(
                    "Parameter max_duration must match the timestamp column dtype"
                )
        if pd.api.types.is_timedelta64_dtype(max_duration):
            if not pd.api.types.is_datetime64_dtype(data[timestamp]):
                raise TypeError(
                    "Parameter max_duration must match the timestamp column dtype"
                )
    if fill_duration:
        if pd.api.types.is_numeric_dtype(fill_duration):
            if not pd.api.types.is_numeric_dtype(data[timestamp]):
                raise TypeError(
                    "Parameter fill_duration must match the timestamp column dtype"
                )
        if pd.api.types.is_timedelta64_dtype(fill_duration):
            if not pd.api.types.is_datetime64_dtype(data[timestamp]):
                raise TypeError(
                    "Parameter fill_duration must match the timestamp column dtype"
                )

    df = data.sort_values(timestamp)

    if direction == "forward":
        duration = -1 * df.groupby(group_by)[timestamp].diff(periods=-1, axis=0)
    elif direction == "backward":
        duration = df.groupby(group_by)[timestamp].diff(periods=1, axis=0)

    if any((duration == 0).fillna(False)):
        raise ValueError("Timestamps must be unique within grouping structure")

    if max_duration:
        duration = duration.clip(upper=max_duration)
    if fill_duration:
        duration = duration.fillna(fill_duration)

    if direction == "forward":
        df[f"{timestamp}_start"] = df[timestamp]
        df[f"{timestamp}_end"] = df[timestamp] + duration
    elif direction == "backward":
        df[f"{timestamp}_start"] = df[timestamp] - duration
        df[f"{timestamp}_end"] = df[timestamp]

    return df


def overlap_to_discrete(
    data: pd.DataFrame,
    timestamp_start: Optional[str | int | float | tuple] = "timestamp_start",
    timestamp_end: Optional[str | int | float | tuple] = "timestamp_end",
    value: Optional[str | int | float | tuple | None] = "value",
    group_by: Optional[list | pd.Series | pd.Grouper | Callable | None] = None,
    keep: Optional[str] = "last",
) -> pd.DataFrame:
    """
    Transforms a dataframe with ranged measurements with overlapping intervals
    to a dataframe with non-overlapping intervals. The value of the new
    intervals is the sum of the overlapping intervals.

    :param data: pandas DataFrame with records of single measurements
    :param timestamp_start: column name of start timestamp
    :param timestamp_end: column name of end timestamp
    :param value: column name of value to transform
    :param group_by: passed directly to pandas.groupby (see pandas documentation)
    :param keep: determine which value to keep: 'first' | 'last'| 'sum'
        last: keep the value of the last interval, sets the first values' end time
         to the start time of the last value
        first: keep the value of the first interval, sets the last values' start time
         to the end time of the first value
        sum: sum the values of all overlapping intervals
    :return:
    """

    passed_columns: list = [timestamp_start, timestamp_end]

    if value:
        passed_columns += [value]

    if group_by:
        if not isinstance(group_by, pd.Series):
            if not callable(group_by):
                if not isinstance(group_by, list):
                    group_by = [group_by]
                if None in group_by:
                    raise ValueError("group_by cannot contain None")
                passed_columns += group_by

    assert_keys_in_object(obj=data.columns, keys=passed_columns)

    _keep_values = ["first", "last", "sum"]
    if keep not in _keep_values:
        raise ValueError(f"parameter keep needs to be any of {'|'.join(_keep_values)}")

    df = data.sort_values(timestamp_start).copy()

    if keep == "last":
        # TODO: this only handles directly adjacent records, but does not handle
        #  multiple overlapping records.
        # TODO: this does not handle records that are completely within another
        #  record
        shifted = df.groupby(group_by).shift(periods=-1)[[timestamp_start, value]]
        df.loc[(df[timestamp_end] > shifted[timestamp_start]), timestamp_end] = shifted[
            timestamp_start
        ]
    elif keep == "first":
        shifted = df.groupby(group_by).shift(periods=1)[[timestamp_end, value]]
        df.loc[
            (df[timestamp_start] < shifted[timestamp_end]), timestamp_start
        ] = shifted[timestamp_end]
    elif keep == "sum":
        # Transform dataframe from records with start and stop times, to
        # individual records of a start time and records of a stop time

        id_vars = [value] + group_by if group_by else [value]

        df_melt = df.melt(
            id_vars=id_vars,
            value_vars=[timestamp_start, timestamp_end, value],
            var_name="_timestamp",
            value_name="_timestamp_value",
        )
        # Inverse the value for records indicating the ending period
        df_melt.loc[df_melt["_timestamp"] == timestamp_end, value] *= -1

        # Group records where timestamps match and sum the values
        by: list = ["_timestamp_value"] + group_by if group_by else ["_timestamp_value"]
        df_sum = (
            df_melt.groupby(
                by=by,
                observed=True,
                sort=False,
                dropna=False,
            )[value]
            .sum()
            .reset_index()
        )
        df_sum.sort_values(by=by, inplace=True)
        df_sum["_cumsum"] = df_sum.groupby(
            by=group_by,
            observed=True,
            sort=False,
            dropna=False,
        )[value].cumsum()

        # transform back to ranged values
        df_sum["_timestamp_value_end"] = df_sum.groupby(
            by=group_by,
            observed=True,
            sort=False,
            dropna=False,
        )["_timestamp_value"].shift(-1)
        df_sum.dropna(subset=["_timestamp_value_end"], inplace=True)
        df_sum.drop(columns=value, inplace=True)
        df_sum.rename(
            columns={
                "_timestamp_value": timestamp_start,
                "_timestamp_value_end": timestamp_end,
                "_cumsum": value,
            },
            inplace=True,
        )

        final_columns: list = list()
        if group_by:
            final_columns += group_by + [timestamp_start, timestamp_end, value]
        else:
            final_columns += [timestamp_start, timestamp_end, value]
        df = df_sum[final_columns]

    return df


def join_adjoining(
    data: pd.DataFrame,
    timestamp_start: Optional[str | int | float | tuple] = "timestamp_start",
    timestamp_end: Optional[str | int | float | tuple] = "timestamp_end",
    values: Optional[list[str | int | float | tuple]] = None,
    group_by: Optional[list[str | int | float | tuple]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Joins adjoining records with the same value to reduce the dataframe size and
     speed up processing. Assumes non-overlapping data for maximal reduction.

    If records have the same value, and the end timestamp of the first record is
    equal to the start timestamp of the next record, then the two records are
    joined. Instead of repeating this process until no more records can be
    merged, we will use a groupby operation to merge all records that can be
    merged in one go. For this, we need to determine which records belong to the
    same group. We will flag all records where the end time matches the start
    time of the next record and the values match as well. We will have to
    include the last record as well if the last record's start time matches the
    previous records end time. For this we will shift the flag one period down
    (+1) to fill empty values.

    We will now have groups of True and False values, where we want to group all
    True values together. We can do this by creating another boolean flag that
    is True when the previous flag is False and the current flag is True. We
    will then take the cumulative sum of this flag, which will give us a unique
    id for each group. We can then group by this id and take the first start
    time and the last end time to get the start and end time of the merged
    records. We can also take the first value, since all values are the same.

    :param data: dataframe of features with start and end times and values
    :param timestamp_start: string of column name
    :param timestamp_end: string of column name
    :param values: list of strings of column names
    :param group_by: list of strings with column names to group by
    :return: pandas.DataFrame of start and end times
    """

    for key, value in kwargs.items():
        match key:
            case "value":
                raise ValueError("Use keyword argument values instead of value")
            case _:
                warnings.warn(f"Unknown parameter {key} with value {value} is ignored")

    passed_columns: list = [timestamp_start, timestamp_end]
    if values is not None:
        passed_columns = passed_columns + values
    if group_by:
        if None in group_by:
            raise ValueError("group_by cannot contain None")
        passed_columns += group_by
    assert_keys_in_object(obj=data.columns, keys=passed_columns)

    sort_by: list = group_by.copy() if group_by else []
    sort_by += [timestamp_start]
    df = data.sort_values(by=sort_by).copy()

    n_groups = dict()
    grouper: list[Any] = list()
    if group_by is not None:
        grouper = grouper + group_by
    if values is not None:
        grouper = grouper + values
    if len(grouper) == 0:
        raise ValueError("group_by and values cannot both be None")

    groups = df.groupby(grouper, sort=False, observed=True, dropna=False).groups
    for i, gr in enumerate(groups.values()):
        for g in gr:
            n_groups[g] = i

    # assign to df because this sorts the index properly for the masking
    # as the dataframe is sorted on time, but the index is usually not reset
    df["_value_groups"] = pd.Series(n_groups.values(), index=n_groups.keys())
    value_groups_shifted = df["_value_groups"].shift(periods=-1)

    df_grouped_shifted = df.groupby(group_by).shift(periods=-1)
    mask_end_is_next_start: pd.Series = (
        df[timestamp_end] == df_grouped_shifted[timestamp_start]
    )
    mask_value_is_next_value: pd.Series = df["_value_groups"] == value_groups_shifted
    match_time_and_value_inverse = ~(mask_end_is_next_start & mask_value_is_next_value)

    match_time_and_value_inverse_shifted = match_time_and_value_inverse.shift(
        periods=1
    ).fillna(True)
    group_id = match_time_and_value_inverse_shifted.cumsum()

    by: list = group_by + [group_id] if group_by else [group_id]
    df_grouped = df.groupby(by=by)

    values_dict = {
        timestamp_start: "first",
        timestamp_end: "last",
    }
    if values is not None:
        for v in values:
            values_dict[v] = "first"

    df_grouped_agg = df_grouped.agg(values_dict)
    df_grouped_agg.reset_index(inplace=True)
    if group_by:
        df_grouped_agg.drop(columns=f"level_{len(group_by)}", inplace=True)
    return df_grouped_agg


def merge_windows(
    windows: pd.DataFrame,
    measurements: pd.DataFrame,
    on: list[str],
    windows_start: str | int | float | tuple,
    windows_end: str | int | float | tuple,
    measurements_start: str | int | float | tuple,
    measurements_end: str | int | float | tuple,
    group_by: list[str | int | float | tuple],
    variable_id: Optional[str | int | float | tuple] = None,
    value: Optional[str | int | float | tuple] = "value",
    value_unit: tuple[float, str] = (1, "h"),
    agg_func: Optional[str] = "mean",
    map_columns: Optional[bool] = True,
    keep_index: Optional[bool] = True,
) -> pd.DataFrame:
    """
    Merges windows with measurements. The measurements are assumed to be
    non-overlapping.

    :param windows: pandas DataFrame with records of single measurements
    :param measurements: pandas DataFrame with records of single measurements
    :param on: columns to merge on
    :param windows_start: column name of start timestamp
    :param windows_end: column name of end timestamp
    :param measurements_start: column name of start timestamp
    :param measurements_end: column name of end timestamp
    :param group_by: passed directly to pandas.groupby (see pandas documentation)
    :param variable_id: column name of variable id,
                         used for pivoting data to wide format
    :param value: column name of value
    :param value_unit: tuple of value unit (e.g. (1, 'h') for 1 x/hr)
    :param agg_func: aggregation function to use for merging
        sum_production: weighted sum of values, assumes default unit is 1 x/hr,
                         adjust value_unit for different rates
        mean - weighted mean of values based on overlap within window
        std - weighted standard deviation for 1D arrays
        min - minimum value within window
        max - maximum value within window
        first - first record in grouping structure, assumes ascending sort
        last - last record in grouping structure, assumes ascending sort

    :param map_columns: cast object columns as categories and restore objects before
                         returning dataframe
    :return:
    """

    # TODO: group_by is used for grouping by, but also for pivoting. If no group_by is
    #  passed, we should still pivot.

    assert group_by, "group_by cannot be None for now as pivots requires it"

    columns_windows: list = list(windows.columns)
    columns_measurements: list = list(measurements.columns)
    columns_both: list = list(set(columns_windows + columns_measurements))
    passed_columns_windows: list = on + [windows_start, windows_end]
    passed_columns_measurements: list = on + [
        measurements_start,
        measurements_end,
        value,
    ]
    passed_columns_both: list = list(
        set(passed_columns_windows + passed_columns_measurements)
    )
    if group_by:
        if None in group_by:
            raise ValueError("group_by cannot contain None")
        passed_columns_both += group_by
    assert_keys_in_object(columns_windows, passed_columns_windows)
    assert_keys_in_object(columns_measurements, passed_columns_measurements)
    assert_keys_in_object(columns_both, passed_columns_both)

    _agg_func_options: list[str] = [
        "mean",
        "std",
        "min",
        "max",
        "first",
        "last",
        "sum_production",
        "any",
        "all",
        "trend",
        "ohe",
    ]
    if agg_func not in _agg_func_options:
        raise ValueError(
            f"Aggregate function {agg_func} not available: choose "
            f"{_agg_func_options.__repr__()}"
        )

    if measurements[value].isnull().any():
        raise ValueError(
            "Measurements contain null values. Impute or drop null values before "
            "merging."
        )

    _windows_length_start = windows.shape[0]

    map_list: list = list()
    if map_columns:
        for col in measurements.columns:
            if measurements[col].dtype == "object":
                measurements[col] = measurements[col].astype("category")
                map_list.append(col)

    __select_columns = [measurements_start, measurements_end, variable_id, value]
    _select_columns = ["w.*"] + [f"m.{x}" for x in __select_columns if x]
    column_clause = ", ".join(_select_columns)
    on_clause = " AND ".join([f"w.{col} = m.{col}" for col in on])
    order_clause = ", ".join([f"w.{col} ASC" for col in on])

    query = f"""
        SELECT {column_clause}
        FROM windows w
        LEFT OUTER JOIN measurements m
        ON {on_clause}
        AND m.{measurements_end} >= w.{windows_start}
        AND m.{measurements_start} < w.{windows_end}
        ORDER BY {order_clause}, w.{windows_start} ASC, m.{measurements_start} ASC
        """

    data = dd.query(query).to_df().reset_index(drop=True)

    window_duration = data[windows_end] - data[windows_start]
    start = data[[measurements_start, windows_start]].max(axis=1)
    end = data[[measurements_end, windows_end]].min(axis=1)
    data["_duration_in_window"] = end - start
    data["_value_weight_factor"] = data["_duration_in_window"] / window_duration

    if agg_func == "sum_production":
        if value_unit == (1, "h"):
            UserWarning(
                "Aggregation function 'sum' assumes the default unit is 1 x/hr (pandas "
                "timedelta)."
            )
        if data["_duration_in_window"].dtype == "timedelta64[ns]":
            data["_duration_in_window"] /= pd.to_timedelta(*value_unit)
        elif data["_duration_in_window"].dtype in ["int", "float"]:
            data["_duration_in_window"] /= value_unit
        else:
            raise ValueError(
                f"Unknown dtype {data['_duration_in_window'].dtype} for column"
                f" '_duration_in_window'"
            )
        weighted_values = data.groupby(group_by).apply(
            lambda x: np.sum(x[value] * x["_duration_in_window"])
        )
    elif agg_func == "min":
        weighted_values = data.groupby(group_by).apply(lambda x: np.min(x[value]))
    elif agg_func == "max":
        weighted_values = data.groupby(group_by).apply(lambda x: np.max(x[value]))
    elif agg_func == "mean":
        weighted_values = data.groupby(group_by).apply(
            lambda x: _np_average(x[value], weights=x["_value_weight_factor"])
        )
    elif agg_func == "std":
        weighted_values = data.groupby(group_by).apply(
            lambda x: np.sqrt(np.cov(x[value], aweights=x["_value_weight_factor"]))
        )
    elif agg_func == "last":
        weighted_values = (
            data.groupby(group_by).tail(1).set_index(group_by)[value].rename(0)
        )
    elif agg_func == "first":
        weighted_values = (
            data.groupby(group_by).head(1).set_index(group_by)[value].rename(0)
        )
    elif agg_func == "any":
        weighted_values = data.groupby(group_by).apply(lambda x: np.any(x[value]))
    elif agg_func == "all":
        weighted_values = data.groupby(group_by).apply(lambda x: np.all(x[value]))
    elif agg_func == "trend":
        # weighted_values = data.groupby(group_by).apply(lambda x: np.polyfit(x[value],
        # x["_duration_in_window"], 1))
        weighted_values = data.groupby(group_by).apply(
            lambda x: _sp_linregress(
                x=x[measurements_start], y=x[value], value_unit=value_unit
            )
        )
    elif agg_func == "ohe":
        dummies = data.groupby(group_by).apply(
            lambda x: pd.get_dummies(
                x[value],
                prefix=x[variable_id].values[0] if x[variable_id].values else None,
            )
        )
        ohe_group_by = list(set(group_by) - set([variable_id]))
        weighted_values = dummies.groupby(ohe_group_by).any(skipna=True)
    else:
        raise ValueError(f"Unknown aggregation function {agg_func}")

    if variable_id is None:
        return_data = windows.merge(
            weighted_values.reset_index(), how="left", on=group_by
        )
    else:
        if group_by:
            index = [x for x in group_by if x != variable_id]
        else:
            index = None

        if agg_func == "ohe":
            return_data = weighted_values.reset_index()
        else:
            return_data = (
                weighted_values.reset_index()
                .pivot(
                    index=index,
                    columns=variable_id,
                    values=0,
                )
                .reset_index()
            )

        # add old columns if more than group by columns were passed
        merge_on = [x for x in group_by if x != variable_id]

        if keep_index:
            windows, names = _save_index(data=windows)
            return_data = windows.merge(return_data, how="left", on=merge_on)
            return_data = _restore_index(data=return_data, names=names)
        else:
            return_data = windows.merge(return_data, how="left", on=merge_on)

    if map_columns:
        for col in map_list:
            if col in return_data.columns:
                return_data[col] = return_data[col].astype("object")

    return_data_length_end = return_data.shape[0]

    if _windows_length_start != return_data_length_end:
        warnings.warn(
            f"Number of rows changed from {_windows_length_start} to "
            f"{return_data_length_end} due to merging."
        )

    return return_data


def _sp_linregress(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    value_unit: tuple[float, "str"],
) -> float:
    """
    A wrapper for scipy linear regression, handling datetime64[ns] conversion to
    numeric values. The resulting slope is expressed as the change in y per unit
    of x, where x is expressed in the unit of value_unit.

    Series with a length of 0 or 1 return np.nan as no (meaningful) slope can be
    calculated.

    :param x:
    :param y:
    :param value_unit:
    :return:
    """
    if len(set(x)) <= 1:
        return np.nan
    if is_datetime64_ns_dtype(x):
        _factor = pd.Timedelta(*value_unit).total_seconds()
        x = x.astype("int64") / 1e9 / _factor
    return linregress(x=x, y=y).slope


def _np_average(a: np.ndarray, *args: Any, **kwargs: Any) -> float | np.ndarray:
    """
    Weighted average of array
    :param a: array
    :param weights: weights
    :return: weighted average
    """
    _weights = kwargs.get("weights", None)
    if _weights is not None:
        if sum(_weights) == 0:
            return np.nan
    ma: np.ma.MaskedArray = np.ma.MaskedArray(a, mask=np.isnan(a))
    return np.average(ma, *args, **kwargs)


def _save_index(data: pd.DataFrame) -> tuple[pd.DataFrame, str | list[str]]:
    _index, _names = _track_index(index=data.index)
    data["__merge_index__"] = _index
    return data, _names


def _restore_index(data: pd.DataFrame, names: str | list[str]) -> pd.DataFrame:
    if isinstance(names, list):
        data.index = pd.MultiIndex.from_tuples(
            data["__merge_index__"],
            names=names,
        )
    else:
        data.set_index("__merge_index__", inplace=True, drop=True)
        data.index.name = names
    return data


def _track_index(
    index: pd.Index | pd.MultiIndex | pd.RangeIndex,
) -> tuple[pd.Series, str | list[str]]:
    if isinstance(index, pd.MultiIndex):
        _values = index.to_flat_index().values
        _names = index.names
    else:
        _values = index.values
        _names = index.name
    _index = pd.Series(_values, name="__merge_index__", index=index)
    return _index, _names


def create_windows(
    data: pd.DataFrame,
    start: str,
    stop: str,
    window_duration: int | float | pd.Timedelta,
    bin_align_start: bool = True,
    bin_keep_remainder: bool = True,
) -> pd.DataFrame:
    """
    Create records for any duration based on provided start and end times

    Create windows of a given duration from a dataframe with start and stop timestamps.
    The start and stop timestamps must be of the same type, either datetime or numeric.
    The passed value for window_duration must be compatible with the type of the window
    start and stop times: either timedelta if start and stop are datetime or numeric if
    start and stop are numeric.

    :param data: pandas DataFrame
    :param start: name of column containing start timestamps
    :param stop: name of column containing stop timestamps
    :param window_duration: duration of the windows as timedelta or numeric
    :param bin_align_start: bool to align windows to start timestamps
    :param bin_keep_remainder: bool to keep remainder of windows that do not fit
    :return: pandas DataFrame with windows
    """

    if "window_id" in data.columns:
        raise AssertionError("'window_id' column is already used, rename this column")
    assert_keys_in_object(obj=data.columns, keys=[start, stop])
    assert_type_compatibility(
        data=data, columns=[start, stop], values=[window_duration]
    )
    assert_stop_after_start(data=data, start=start, stop=stop, allow_empty_values=False)

    list_of_dicts: list[dict] = list()

    outer_j: int = 0

    for record in data.itertuples():
        rec: dict = record._asdict()

        traj_dict: dict = {c: list() for c in data.columns}
        traj_dict.update({"window_id": list()})

        start_value = rec.get(start, None)
        stop_value = rec.get(stop, None)
        if start_value is None or stop_value is None:
            raise ValueError(
                f"start or stop value is None: {start_value}, {stop_value}"
            )
        else:
            traj_duration = stop_value - start_value

        if bin_keep_remainder:
            n_windows = np.ceil(traj_duration / window_duration).astype(int)
        else:
            n_windows = np.floor(traj_duration / window_duration).astype(int)

        for j in range(n_windows):
            for k in traj_dict.keys():
                if k == "window_id":
                    traj_dict["window_id"].append(outer_j)
                elif k == start:
                    if bin_align_start:
                        traj_dict[start].append(start_value + j * window_duration)
                    else:
                        start_value_to_append = (
                            stop_value - (n_windows - j) * window_duration
                        )
                        if start_value_to_append < start_value:
                            start_value_to_append = start_value
                        traj_dict[start].append(start_value_to_append)
                elif k == stop:
                    if bin_align_start:
                        stop_value_to_append = start_value + (j + 1) * window_duration
                        if stop_value_to_append > stop_value:
                            stop_value_to_append = stop_value
                        traj_dict[stop].append(stop_value_to_append)
                    else:
                        stop_value_to_append = (
                            stop_value - (n_windows - j - 1) * window_duration
                        )
                        traj_dict[stop].append(stop_value_to_append)
                else:
                    traj_dict[k].append(rec.get(k))
            outer_j += 1
        list_of_dicts.append(traj_dict)

    full_dict: dict = {c: list() for c in list_of_dicts[0].keys()}
    for traj_dict in list_of_dicts:
        for k, v in traj_dict.items():
            for i in v:
                full_dict[k].append(i)

    return pd.DataFrame(full_dict)
