from utils.logger import setup_logger
from typing import TypedDict, Literal
import pandas as pd


logger = setup_logger(__name__)


class ColumnDefinition(TypedDict):
    name: str
    type: Literal["numeric", "nominal", "ordinal"]


def build_preprocess_dict(df: pd.DataFrame, columns_list: list[ColumnDefinition]):
    """
    Constructs a preprocessing configuration dictionary based on column metadata and data.

    Parameters:
        columns_list (list): List of dictionaries with at least 'name' and 'type' keys.
        df (pandas.DataFrame): The dataset containing the actual column values.

    Returns:
        dict: Dictionary with keys:
              - numeric_cols - list of column names of numerical features
              - categorical_cols - list of column names of categorical features
              - ordinal_cols - a dictionary where each key is a column name, each value is a list of ordinal_values (ordered)
              - y_col: always empty (unsupervised)
    """

    numeric_cols = []
    categorical_cols = []
    ordinal_cols = {}

    for col in columns_list:
        col_name = col.get("name")
        col_type = col.get("type")

        if not col_name or not col_type:
            logger.warning(f"[AutoML] Skipping malformed column entry: {col}")
            continue

        if col_type == "numeric":
            numeric_cols.append(col_name)
        elif col_type == "nominal":
            categorical_cols.append(col_name)
        elif col_type == "ordinal":
            try:
                unique_vals = sorted(df[col_name].dropna().unique().tolist())
                ordinal_cols[col_name] = unique_vals
            except Exception as e:
                logger.warning(f"[AutoML] Couldn't extract ordinal values for '{col_name}': {e}")

        else:
            logger.warning(f"[AutoML] Unknown column type for '{col_name}': '{col_type}' – skipping.")

    preprocess_dict = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "ordinal_cols": ordinal_cols,
        "y_col": [],  # No label column in unsupervised clustering
    }

    logger.debug(f"[AutoML] Built preprocess_dict: {preprocess_dict}")
    return preprocess_dict
