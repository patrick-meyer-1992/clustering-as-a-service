import pandas as pd
from workers.fit_config import prepare_fit_params

def test_prepare_fit_params_returns_valid_dict():
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': ['x', 'y', 'z']
    })
    columns = ['a', 'b', 'c']

    fit_params = prepare_fit_params(df, columns, ['KMeans'], ['PCA'], ['silhouetteScore'], 30, 10)

    assert isinstance(fit_params, dict)
    assert "X" in fit_params
    assert "clustering_algorithms" in fit_params
    assert fit_params["cutoff_time"] == 30
    assert fit_params["n_evaluations"] == 10