"""Tests for data_ingestion_utils.py."""

import pytest
import pandas as pd
import numpy as np

from src.DynamicPricingEngine.utils.data_ingestion_utils import (
    leap_year,
    time_subtract,
    dtype_downcast,
)


class TestLeapYear:
    def test_leap_year_divisible_by_400(self):
        assert leap_year(2000) is True
        assert leap_year(2400) is True

    def test_leap_year_divisible_by_4_not_100(self):
        assert leap_year(2024) is True
        assert leap_year(2020) is True

    def test_not_leap_year_divisible_by_100_not_400(self):
        assert leap_year(2100) is False
        assert leap_year(1900) is False

    def test_not_leap_year_odd(self):
        assert leap_year(2023) is False
        assert leap_year(2025) is False


class TestTimeSubtract:
    def test_january(self):
        assert time_subtract("2024-01-15") == 31

    def test_february_non_leap(self):
        assert time_subtract("2023-02-15") == 28

    def test_february_leap(self):
        assert time_subtract("2024-02-15") == 29

    def test_april(self):
        assert time_subtract("2024-04-15") == 30

    def test_december(self):
        assert time_subtract("2024-12-01") == 31


class TestDtypeDowncast:
    def test_downcast_integers(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1000, 2000, 3000]})
        result = dtype_downcast(df.copy())
        assert result["a"].dtype in (np.int8, np.int16, np.int32, np.int64)
        assert result["b"].dtype in (np.int16, np.int32, np.int64)

    def test_downcast_floats(self):
        df = pd.DataFrame({"a": [1.5, 2.5, 3.5]})
        result = dtype_downcast(df.copy())
        assert result["a"].dtype in (np.float32, np.float64)

    def test_object_to_category(self):
        df = pd.DataFrame({"a": ["x", "y", "x", "y"]})
        result = dtype_downcast(df.copy())
        assert result["a"].dtype.name == "category"

    def test_no_crash_on_empty(self):
        df = pd.DataFrame()
        result = dtype_downcast(df)
        assert result.empty

    def test_reduces_memory(self):
        large = pd.DataFrame({
            "id": range(10000),
            "value": [float(i) for i in range(10000)],
            "label": ["cat" if i % 2 == 0 else "dog" for i in range(10000)],
        })
        before = large.memory_usage(deep=True).sum()
        result = dtype_downcast(large)
        after = result.memory_usage(deep=True).sum()
        assert after <= before
