from utils.logger import setup_logger

logger = setup_logger(__name__)


def build_preprocess_dict(columns_list):
    """
    Constructs a preprocessing configuration dictionary based on column metadata.

    This function parses a list of column definitions and groups them by type
    into separate lists for numeric, nominal (categorical), and ordinal features.
    It also initializes an empty target column list, since clustering is unsupervised.

    Parameters:
        columns_list (list): A list of dictionaries, each containing at least the keys:
            - "name" (str): The column name.
            - "type" (str): The column type, one of {"numeric", "nominal", "ordinal"}.

    Returns:
        dict: A dictionary with the structure:
            {
                "numeric_cols": list of str,
                "categorical_cols": list of str,
                "ordinal_cols": list of str,
                "y_col": []  # always empty in unsupervised clustering
            }

    Notes:
        - Invalid or malformed column entries are skipped with a warning.
        - Unknown types are ignored and not included in the result.
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
        "y_col": [],  # No label column in unsupervised clustering
    }

    logger.debug(f"[Preprocessing] Built preprocess_dict: {preprocess_dict}")
    return preprocess_dict
