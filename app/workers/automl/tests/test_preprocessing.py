import pytest
import pandas as pd
from workers.automl.fit_config import build_preprocess_dict

def test_build_preprocess_dict_basic():
    data = {
        "income": [3000, 4000, 5000],
        "age": [25, 30, 35],
        "education": ["High School", "Bachelor", "Master"],
        "satisfaction": [1, 2, 3],
    }
    df = pd.DataFrame(data)

    columns = [
        {"name": "income", "type": "numeric"},
        {"name": "age", "type": "numeric"},
        {"name": "education", "type": "nominal"},
        {"name": "satisfaction", "type": "ordinal"},
    ]

    expected = {
        "numeric_cols": ["income", "age"],
        "categorical_cols": ["education"],
        "ordinal_cols": {"satisfaction": [1, 2, 3]},
        "y_col": [],
    }

    result = build_preprocess_dict(df, columns)
    assert result == expected


def test_build_preprocess_dict_ordinal_with_nan():
    df = pd.DataFrame({"satisfaction": [1, 2, None, 3]})
    columns = [{"name": "satisfaction", "type": "ordinal"}]
    result = build_preprocess_dict(df, columns)
    assert result["ordinal_cols"]["satisfaction"] == [1.0, 2.0, 3.0]


def test_build_preprocess_dict_empty_df():
    df = pd.DataFrame(columns=["a", "b", "c"])
    columns = [
        {"name": "a", "type": "numeric"},
        {"name": "b", "type": "nominal"},
        {"name": "c", "type": "ordinal"},
    ]
    result = build_preprocess_dict(df, columns)
    assert result == {
        "numeric_cols": ["a"],
        "categorical_cols": ["b"],
        "ordinal_cols": {"c": []},
        "y_col": [],
    }


def test_build_preprocess_dict_unknown_ordinal_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    columns = [{"name": "not_in_df", "type": "ordinal"}]

    result = build_preprocess_dict(df, columns)
    assert result == {
        "numeric_cols": [],
        "categorical_cols": [],
        "ordinal_cols": {},  # Es sollte leergelassen werden
        "y_col": [],
    }


def test_build_preprocess_dict_empty_columns():
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = build_preprocess_dict(df, [])
    assert result == {
        "numeric_cols": [],
        "categorical_cols": [],
        "ordinal_cols": {},
        "y_col": [],
    }


