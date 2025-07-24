import pandas as pd

from ..fit_config import prepare_fit_params, validate_clustering_num


def test_prepare_fit_params_returns_valid_dict():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})

    columns = [
        {"name": "a", "type": "numeric"},
        {"name": "b", "type": "numeric"},
        {"name": "c", "type": "nominal"},
    ]

    fit_params = prepare_fit_params(df, columns, ["KMeans"], ["PCA"], ["silhouetteScore"], 30, 10, None)

    assert isinstance(fit_params, dict)
    assert fit_params["df"].equals(df)
    assert fit_params["cluster_alg_ls"] == ["KMeans"]
    assert fit_params["dim_reduction_alg_ls"] == ["PCA"]
    assert fit_params["cutoff_time"] == 30
    assert fit_params["n_evaluations"] == 10


def test_prepare_fit_params_returns_NullModel_on_null():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})

    columns = [
        {"name": "a", "type": "numeric"},
    ]

    fit_params = prepare_fit_params(df, columns, ["KMeans"], None, ["silhouetteScore"], 30, 10, None)

    assert isinstance(fit_params, dict)
    assert fit_params["dim_reduction_alg_ls"] == ["NullModel"]


def test_prepare_fit_params_returns_NullModel_on_empty_list():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})

    columns = [
        {"name": "a", "type": "numeric"},
    ]

    fit_params = prepare_fit_params(df, columns, ["KMeans"], [], ["silhouetteScore"], 30, 10, None)

    assert isinstance(fit_params, dict)
    assert fit_params["dim_reduction_alg_ls"] == ["NullModel"]


def test_prepare_fit_params_returns_NullModel_on_empty_string_in_list():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})

    columns = [
        {"name": "a", "type": "numeric"},
    ]

    fit_params = prepare_fit_params(df, columns, ["KMeans"], [""], ["silhouetteScore"], 30, 10, None)

    assert isinstance(fit_params, dict)
    assert fit_params["dim_reduction_alg_ls"] == ["NullModel"]



def test_prepare_fit_params_returns_NullModel_on_empty_string():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["x", "y", "z"]})

    columns = [
        {"name": "a", "type": "numeric"},
    ]

    fit_params = prepare_fit_params(df, columns, ["KMeans"], "", ["silhouetteScore"], 30, 10, None)

    assert isinstance(fit_params, dict)
    assert fit_params["dim_reduction_alg_ls"] == ["NullModel"]


def test_validate_clustering_num():
    assert validate_clustering_num((2, 5)) == (2, 5)
    assert validate_clustering_num((5, 2)) is None
    assert validate_clustering_num("abc") is None
    assert validate_clustering_num((2, "a")) is None


    #    "df": df,
    #    "cluster_alg_ls": clustering_algorithms,
    #    "dim_reduction_alg_ls": dim_reduction_algorithms,
    #    "optimizer": "smac",
    #    "n_evaluations": n_evaluations,
    #    "run_obj": "quality",
    #    "seed": 27,
    #    "cutoff_time": cutoff_time,
    #    "preprocess_dict": preprocessing_dict,
    #    "evaluator": get_evaluator(evaluator_ls, weights=[1, 1, 1], clustering_num=None, min_proportion=0.01),
    #    "n_folds": 3,
    #   "warmstart": False,
    #   "general_metafeatures": MetafeatureMapper.getGeneralMetafeatures(),
    #   "numeric_metafeatures": MetafeatureMapper.getNumericMetafeatures(),
    #   "categorical_metafeatures": [],
    #    "verbose_level": 1,