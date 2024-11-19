import numpy as np
import pandas as pd
import pytest

from tadam.assertions import (
    _check_type_compatibility,
    assert_keys_in_object,
    assert_type_compatibility,
)


def test_assert_keys_in_object():
    obj: list = ["a", "b", 1, 0.3, (1, "2")]
    list_of_items_pass: list = ["a", "b", 1, 0.3, (1, "2")]
    list_of_items_fail: list = ["a", "b", 1, 0.3, 0.4]

    assert assert_keys_in_object(obj=obj, keys=list_of_items_pass) is None
    with pytest.raises(KeyError):
        assert_keys_in_object(obj=obj, keys=list_of_items_fail)


def test__check_type_compatibility():
    """
    Test if the function correctly identifies the type of the input and
    returns the correct boolean value with exceptions for datetime and timedelta
    """

    assert _check_type_compatibility(
        type_1=type(1), type_2=type(2), datetime_matches_timedelta=False
    )
    assert _check_type_compatibility(
        type_1=pd.Timestamp, type_2=pd.Timestamp, datetime_matches_timedelta=False
    )
    assert _check_type_compatibility(
        type_1=pd.Timedelta, type_2=pd.Timedelta, datetime_matches_timedelta=False
    )
    assert _check_type_compatibility(
        type_1=pd.Timestamp, type_2=pd.Timedelta, datetime_matches_timedelta=True
    )

    with pytest.raises(AssertionError):
        # string type matching not implemented as this function focuses on calculations
        assert _check_type_compatibility(
            type_1=type("1"), type_2=type("two"), datetime_matches_timedelta=False
        )
        assert _check_type_compatibility(
            type_1=type("1"), type_2=type("two"), datetime_matches_timedelta=True
        )
        assert _check_type_compatibility(
            type_1=type(pd.to_datetime("2020-01-01")),
            type_2=type(pd.to_datetime("2020-01-02")),
            datetime_matches_timedelta=True,
        )
        assert _check_type_compatibility(
            type_1=type(pd.to_timedelta(1, unit="d")),
            type_2=type(pd.to_timedelta(2, unit="d")),
            datetime_matches_timedelta=True,
        )
        assert _check_type_compatibility(
            type_1=type(np.timedelta64(1, "D")),
            type_2=type(pd.to_timedelta(2, unit="d")),
            datetime_matches_timedelta=True,
        )
        assert _check_type_compatibility(
            type_1=type(np.timedelta64(1, "D")),
            type_2=type(np.datetime64("1970-01-01")),
            datetime_matches_timedelta=False,
        )
        assert _check_type_compatibility(
            type_1=type(pd.to_datetime("2020-01-01")),
            type_2=type(1),
            datetime_matches_timedelta=True,
        )
        assert _check_type_compatibility(
            type_1=type(1),
            type_2=type(pd.to_timedelta(2, unit="d")),
            datetime_matches_timedelta=True,
        )

    assert _check_type_compatibility(
        type_1=type(pd.to_datetime("2020-01-01")),
        type_2=type(pd.to_timedelta(1, unit="d")),
        datetime_matches_timedelta=True,
    )


def test_assert_type_compatibility():
    test_data = pd.DataFrame(
        {
            "start": ["2020-01-01", "2020-01-01", "2020-01-01"],
            "end": ["2020-01-02", "2020-01-02", "2020-01-02"],
            "value": [1, 2, 3],
        }
    )
    test_data["start"] = pd.to_datetime(test_data["start"])
    test_data["end"] = pd.to_datetime(test_data["end"])
    test_data_num = pd.DataFrame(
        {
            "start": [1, 2, 3],
            "end": [4, 5, 6],
            "value": [1, 2, 3],
        }
    )

    with pytest.raises(TypeError):
        assert_type_compatibility(
            data=test_data, columns=["start", "end"], values=[5, 6]
        )

        assert_type_compatibility(
            data=test_data,
            columns=["start", "end"],
            values=[pd.Timedelta(1, "d"), np.datetime64("2020-01-01")],
        )

        # values as pd.Timedelta and np.timedelta64 should be compatible, but are not
        # due to the way the function is written
        assert_type_compatibility(
            data=test_data,
            columns=["start", "end"],
            values=[pd.Timedelta(1, "d"), np.timedelta64(1, "D")],
        )
    assert assert_type_compatibility(
        data=test_data,
        columns=["start", "end"],
        values=[pd.Timedelta(1, "d"), pd.Timedelta(2, "d")],
    )
    assert assert_type_compatibility(
        data=test_data_num, columns=["start", "end"], values=[2, 5.0]
    )
