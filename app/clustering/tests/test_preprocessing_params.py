import pytest
from pydantic import ValidationError

from app.clustering.preprocessing_params import PreProcessingParams


def test_valid_preprocessing_params():
    params = PreProcessingParams(
        scaler="standard",
        use_normalization=True,
        normalization_type="l1",
        use_pca=True,
        pca_components=5,
        imputation_strategy="mean",
    )
    assert params.scaler == "standard"
    assert params.use_pca is True


def test_invalid_scaler():
    with pytest.raises(ValidationError):
        PreProcessingParams(scaler="invalid_scaler")


def test_invalid_normalization_type():
    with pytest.raises(ValidationError):
        PreProcessingParams(use_normalization=True, normalization_type="foo")


def test_pca_components_must_be_positive():
    with pytest.raises(ValidationError):
        PreProcessingParams(use_pca=True, pca_components=0)


def test_default_params_values():
    params = PreProcessingParams()
    assert params.scaler == "auto"
    assert params.use_normalization is False
    assert params.normalization_type == "l2"
    assert params.pca_components == 10


def test_none_allowed_fields():
    params = PreProcessingParams(
        imputation_strategy=None, transform_type=None, outlier_removal=None, feature_selection=None
    )
    assert params.imputation_strategy is None


def test_model_dump_works():
    params = PreProcessingParams()
    dumped = params.model_dump()
    assert isinstance(dumped, dict)
    assert "scaler" in dumped
