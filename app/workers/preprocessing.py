import logging
from workers.logger import setup_logger

logger = setup_logger(__name__)

def build_preprocess_dict(columns_list):
    """
    Builds a preprocess_dict from a list of dictionaries with structure:
    [{"name": "sepal_length", "type": "numeric"}, ...]

    Returns a dictionary with keys:
    - numeric_cols
    - categorical_cols (for type 'nominal')
    - ordinal_cols
    - y_col (always empty for unsupervised clustering)

    :param columns_list: List of dicts with 'name' and 'type' keys
    :return: Dictionary in the expected preprocess_dict format
    """
    numeric_cols = []
    categorical_cols = []
    ordinal_cols = []

    for col in columns_list:
        col_name = col.get("name")
        col_type = col.get("type")

        if not col_name or not col_type:
            logger.warning(f"[Preprocessing] Skipping malformed column entry: {col}")
            continue

        if col_type == "numeric":
            numeric_cols.append(col_name)
        elif col_type == "nominal":
            categorical_cols.append(col_name)
        elif col_type == "ordinal":
            ordinal_cols.append(col_name)
        else:
            logger.warning(f"[Preprocessing] Unknown column type for '{col_name}': '{col_type}' – skipping.")

    preprocess_dict = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "ordinal_cols": ordinal_cols,
        "y_col": []  # No label column in unsupervised clustering
    }

    logger.debug(f"[Preprocessing] Built preprocess_dict: {preprocess_dict}")
    return preprocess_dict


# TODO: Optional – Validate column types against allowed values (numeric, nominal, ordinal)
#       Raise or collect warnings if invalid types are found
