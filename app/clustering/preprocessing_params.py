from typing import Literal

from pydantic import BaseModel, Field


class PreProcessingParams(BaseModel):
    scaler: Literal["auto", "standard", "minmax", "robust", "maxabs"] = Field(
        title="The type of scaler to use for preprocessing",
        default="auto",
        description="Choose 'auto' to let the algorithm decide the best scaler based on the dataset.",
    )
    use_normalization: bool = Field(
        title="Whether to apply normalization to the data",
        default=False,
        description="Set to True to apply normalization.",
    )
    normalization_type: Literal["l1", "l2", "max"] = Field(
        title="The type of normalization to apply",
        default="l2",
        description="Choose 'l1', 'l2' or 'max' for the normalization type.",
    )
    use_pca: bool = Field(
        title="Whether to apply PCA for dimensionality reduction",
        default=False,
        description="Set to True to apply PCA.",
    )
    pca_components: int = Field(
        title="Number of components to keep after PCA",
        default=10,
        description="Choose the number of components to keep after PCA.",
    )
    transform_type: Literal["quantile", "power", None] = Field(
        title="The type of transformation to apply to the data",
        default=None,
        description="Choose 'quantile', 'power' or None for no transformation.",
    )
    imputation_strategy: Literal["mean", "median", "none"] = Field(
        title="Strategy for handling missing values",
        default="mean",
        description="Choose 'mean', 'median' or 'none' for the imputation strategy.",
    )
    outlier_removal: Literal["none", "zscore", "iqr"] = Field(
        title="Method for outlier removal",
        default="none",
        description="Choose 'none', 'zscore' or 'iqr' for the outlier removal method.",
    )
    outlier_threshold: float = Field(
        title="Threshold for outlier detection",
        default=3.0,
        description="Choose the threshold for outlier detection. Only used if outlier_removal is 'zscore'.",
    )
    feature_selection: Literal["none", "low_variance", "constant"] = Field(
        title="Method for feature selection",
        default="none",
        description="Choose 'none', 'low_variance' or 'constant' for the feature selection method.",
    )
    variance_threshold: float = Field(
        title="Variance threshold for feature selection",
        default=0.0,
        description="Choose the variance threshold for feature selection. Only used if "
        "feature_selection is 'low_variance'.",
    )
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "scaler": "auto",
                    "use_normalization": False,
                    "normalization_type": "l2",
                    "use_pca": False,
                    "pca_components": 10,
                    "transform_type": None,
                    "imputation_strategy": "none",
                    "outlier_removal": "none",  # "none", "zscore", "iqr"
                    "outlier_threshold": 3.0,  # only for zscore
                    "feature_selection": "none",  # "none", "low_variance", "constant"
                    "variance_threshold": 0.0,  # for low_variance
                }
            ]
        }
    }
