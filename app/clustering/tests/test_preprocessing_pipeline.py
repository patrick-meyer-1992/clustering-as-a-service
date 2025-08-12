import numpy as np
import pytest

from app.clustering.preprocessing_params import PreProcessingParams
from app.clustering.preprocessing_pipeline import PreprocessingPipeline


def test_scale_step_standard():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    params = PreProcessingParams(scaler="standard")
    pipeline = PreprocessingPipeline(params)
    out = pipeline._scale(X)
    assert out.shape == X.shape
    assert np.allclose(out.mean(axis=0), 0, atol=1e-7)


def test_scale_step_auto_high_variance():
    X = np.array([[1, 2], [1000, 2000], [500, 1000]])
    params = PreProcessingParams(scaler="auto")
    pipeline = PreprocessingPipeline(params)
    out = pipeline._scale(X)
    assert out.shape == X.shape


def test_normalize_step_l2():
    X = np.array([[3, 4]])
    params = PreProcessingParams(use_normalization=True, normalization_type="l2")
    pipeline = PreprocessingPipeline(params)
    out = pipeline._normalize(X)
    assert np.allclose(np.linalg.norm(out), 1.0)


def test_impute_step_mean():
    X = np.array([[1, 2], [np.nan, 4]])
    params = PreProcessingParams(imputation_strategy="mean")
    pipeline = PreprocessingPipeline(params)
    out = pipeline._impute(X)
    assert not np.isnan(out).any()


def test_remove_outliers_zscore():
    X = np.vstack([np.random.normal(0, 1, size=(98, 2)), [100, 100], [-100, -100]])
    params = PreProcessingParams(outlier_removal="zscore", outlier_threshold=3.0)
    pipeline = PreprocessingPipeline(params)
    X_proc = pipeline._remove_outliers(X)
    assert X_proc.shape[0] < X.shape[0]


def test_remove_outliers_iqr():
    X = np.vstack([np.random.normal(0, 1, size=(98, 2)), [20, 20]])
    params = PreProcessingParams(outlier_removal="iqr")
    pipeline = PreprocessingPipeline(params)
    X_proc = pipeline._remove_outliers(X)
    assert X_proc.shape[0] < X.shape[0]


def test_feature_selection_constant():
    X = np.ones((10, 3))  # All columns constant
    params = PreProcessingParams(feature_selection="constant")
    pipeline = PreprocessingPipeline(params)
    with pytest.raises(ValueError, match="No feature in X meets the variance threshold"):
        pipeline._feature_select(X)


def test_feature_selection_low_variance():
    X = np.hstack([np.random.rand(100, 2), np.ones((100, 1))])  # Third column = constant
    params = PreProcessingParams(feature_selection="low_variance", variance_threshold=0.01)
    pipeline = PreprocessingPipeline(params)
    out = pipeline._feature_select(X)
    assert out.shape[1] == 2


def test_reduce_dim_with_pca():
    X = np.random.rand(100, 5)
    params = PreProcessingParams(use_pca=True, pca_components=2)
    pipeline = PreprocessingPipeline(params)
    out = pipeline._reduce_dim(X)
    assert out.shape == (100, 2)


def test_reduce_dim_too_few_dims():
    X = np.random.rand(100, 2)
    params = PreProcessingParams(use_pca=True, pca_components=5)
    pipeline = PreprocessingPipeline(params)
    out = pipeline._reduce_dim(X)
    assert out.shape == X.shape  # PCA not applied


def test_post_transform_quantile():
    X = np.random.rand(100, 2)
    params = PreProcessingParams(transform_type="quantile")
    pipeline = PreprocessingPipeline(params)
    out = pipeline._post_transform(X)
    assert out.shape == X.shape


def test_post_transform_power():
    X = np.random.rand(100, 2)
    params = PreProcessingParams(transform_type="power")
    pipeline = PreprocessingPipeline(params)
    out = pipeline._post_transform(X)
    assert out.shape == X.shape


def test_fit_transform_entire_pipeline():
    X = np.vstack([np.random.normal(0, 1, size=(97, 4)), [np.nan, np.nan, np.nan, np.nan], [1000, 1000, 1000, 1000]])
    params = PreProcessingParams(
        scaler="standard",
        use_normalization=True,
        normalization_type="l2",
        imputation_strategy="mean",
        outlier_removal="zscore",
        feature_selection="low_variance",
        variance_threshold=0.01,
        use_pca=True,
        pca_components=2,
        transform_type="power",
    )
    pipeline = PreprocessingPipeline(params)
    X_proc = pipeline.fit_transform(X)
    assert X_proc.shape[1] == 2


def test_pipeline_with_empty_input():
    X = np.empty((0, 5))
    params = PreProcessingParams()
    pipeline = PreprocessingPipeline(params)
    with pytest.raises(ValueError, match="Found array with 0 sample"):
        pipeline.fit_transform(X)
