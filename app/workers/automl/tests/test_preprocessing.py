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