# tadam
_frequently used functions in medical data science_

## Background
Medical tabular data is often presented in a wide variety of structures, requiring extensive data wrangling. As data
collections increase in size, processing time may scale exponentially. The functions present in this package
were developed to simplify and standardize the processing of medical data.

## How to use
```
from tadam.data_processing import (
    single_to_range,
    overlap_to_discrete,
    join_adjoining,
    impute_missing,
    create_windows,
    merge_windows,
)
```

### Data extraction
After obtaining the data of interest, make sure each table has at least the following columns:
- a patient identifier [object | int]
- a date and time stamp [pd.Datetime64 | int]
- a measurement identifier [object] (e.g. po2, Hb etc.)
- a value [object, int, float, bool] (e.g. 10, 0, -20, True, positive)

```
|index|patient_id|timestamp|measurement|value|
|-----|----------|---------|-----------|-----|
|    0|         0|        0|        po2|   20|
|    1|         1|       33|         hb|    5|
... 
```

### Transformation
#### Single to Range
In the case of settings or medication, measurements are started a start timestamp and stopped at a stop timestamp. 
Measurements at a single timestamp like lab collection only have the timestamp of collection, however, we typically
regard these values to be valid measurements for a certain period of time after collection. In this step, we're defining
how long each single timestamp measurement should be seen as a valid measurement.

For example: each pO2 measurement is valid for 6 hours, but each sodium measurement is valid for 24 hours. For now, this
requires two steps in processing: one for 6 hours, and one for 24 hours. If two measurements follow each other, the
timestamp of the last measurement is used as the stop timestamp for the previous, so no overlap will occur. However,
if these measurements are too far apart, you can define a maximum duration (e.g. 6 hours) to be used. If no next record
is found, the fill_duration keyword will be used. These keywords will typically have the same value, but some exceptions
may occur. One such exception may be values registered every hour, but having slight variation in the exact registration
time where sometimes a value will reported a few minutes later. In this case, you set the maximum duration to 1.5 hours
while setting the fill_duration to 1 hour (or even shorter) depending on the use case. 

```
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
```

```
df2 = single_to_range(
    data=df,
    timestamp="timestamp",
    group_by=["patient_id", "measurement"],
    direction="forward",
    max_duration=pd.Timedelta(6, 'hours')
    fill_duration=pd.Timedelta(6, 'hours')
)

|index|patient_id|timestamp_start|timestamp_end|measurement|value|
|-----|----------|---------------|-------------|-----------|-----|
|    0|         0|              0|            6|        po2|   20|
|    1|         1|             33|           39|         hb|    5|
...
```

#### Overlap to discrete
Some records will overlap when they should not (e.g. lab values, ventilator settings which exclude each other). In this
step, we're going to define which record to keep. You can choose to keep the first value or the last value, or to sum
these values if the values are all ints or floats. Note: for now, this step needs to be done separately for each 
keep option: e.g. all 'first' variables together, then all 'last', then all 'sum', and then concatenate the resulting
dataframes together.

```
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
```

```
df3 = overlap_to_discrete(
    data=df2,
    timestamp_start="timestamp_start",
    timestamp_end="timestamp_end",
    value="value",
    group_by=["patient_id", "measurement"],
    keep="last"
)
```

#### Join adjoining
With data being registered every hour (or even every minute), we may have millions of records where the values are
exactly the same. If data is entered every minute, and the value of, for example, FiO2 remains 50% for even 1 hour,
we can reduce the dataframe size by 59 records for each patient. Doing this on a dataset of 10k patients, with 20 
measurements, the dataframe can be reduced by 11.8m records and is strongly advised to benefit from faster processing
in the next steps.

```
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
```

#### Create windows
In this function, we create a dataframe with set start and end periods called windows. For example, if a patient has
been mechanically ventilated for 3 weeks, but we want to aggregate measurements every hour, we can create a dataframe
which takes every hour within the 3-week period. As the entire period is not always fully divisible by the window
duration, we can choose to align the windows with the start of the period (default) and we can choose to keep the
remainder of the last window which would be longer than the initial period (default). 

```
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
```

```
df: pd.DataFrame [patient_id, start, end]
df2 = create_windows(data=df, start='start', stop='end', window_duration=pd.Timedelta(1, 'hour'))
output: pd.DataFrame [patient_id, window_id, start, end]
```
#### Merge windows
The merge windows function aggregates data over the set periods in the windows dataframe.

```
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
```

#### Impute missing
Merging windows with measurements may result in empty windows for some data which is not missing at random, for example
medication data. If medication records are missing, this usually means no medication was given and these values may
want to be imputed as 0. This function allows for setting a fixed value for each window not already covered. 



```
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
```





