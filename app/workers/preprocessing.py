

def build_preprocess_dict(columns_list):
    """
    Builds a preprocess_dict from a list of dictionaries with structure:
    [{"name": "sepal_length", "type": "numeric"}, ...]

    Returns a dictionary with keys:
    - numeric_cols
    - categorical_cols (for type 'nominal')
    - ordinal_cols
    - y_col (always empty)

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
            print(f"[WARN] Skipping malformed column entry: {col}")
            continue

        if col_type == "numeric":
            numeric_cols.append(col_name)
        elif col_type == "nominal":
            categorical_cols.append(col_name)
        elif col_type == "ordinal":
            ordinal_cols.append(col_name)
        else:
            print(f"[WARN] Unknown column type for '{col_name}': '{col_type}' – skipping.")

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "ordinal_cols": ordinal_cols,
        "y_col": []  # No label column in unsupervised clustering
    }

#TODO: Column checks


