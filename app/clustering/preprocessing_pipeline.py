from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from .preprocessing_params import PreProcessingParams


class PreprocessingPipeline:
    """Reusable preprocessing pipeline.

    Parameters
    ----------
    params : PreProcessingParams
        Configuration object that specifies every preprocessing option.
    """

    def __init__(self, params: PreProcessingParams):
        self.params = params

    # Optional: keep fitted state for transform-only use
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the pipeline on X and return processed array."""
        X_proc = self._scale(X)
        X_proc = self._normalize(X_proc)
        X_proc = self._impute(X_proc)
        X_proc = self._remove_outliers(X_proc)
        X_proc = self._feature_select(X_proc)
        X_proc = self._reduce_dim(X_proc)
        X_proc = self._post_transform(X_proc)
        return X_proc

    # --- step helpers -----------------------------------------------------
    def _scale(self, X: np.ndarray) -> np.ndarray:
        p = self.params
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "maxabs": MaxAbsScaler(),
            "auto": RobustScaler()
            if (pd.DataFrame(X).max() - pd.DataFrame(X).min()).max() > 1000
            else StandardScaler(),
        }
        try:
            scaler = scaler_map[p.scaler]
        except KeyError:
            raise ValueError(f"Unsupported scaler type: {p.scaler}")
        return scaler.fit_transform(X)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if not self.params.use_normalization:
            return X
        norm = Normalizer(norm=self.params.normalization_type)
        return norm.fit_transform(X)

    def _impute(self, X: np.ndarray) -> np.ndarray:
        strategy = self.params.imputation_strategy
        if strategy is None:
            return X
        df = pd.DataFrame(X)
        if strategy == "mean":
            return df.fillna(df.mean()).to_numpy()
        if strategy == "median":
            return df.fillna(df.median()).to_numpy()
        raise ValueError(f"Unsupported imputation strategy: {strategy}")

    def _remove_outliers(self, X: np.ndarray) -> np.ndarray:
        strategy = self.params.outlier_removal
        thr = self.params.outlier_threshold
        if strategy is None:
            return X
        if strategy == "zscore":
            z = np.abs((X - X.mean(0)) / X.std(0))
            return X[(z < thr).all(1)]
        if strategy == "iqr":
            q1, q3 = np.percentile(X, [25, 75], axis=0)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = ((lower <= X) & (upper >= X)).all(1)
            return X[mask]
        raise ValueError(f"Unsupported outlier removal: {strategy}")

    def _feature_select(self, X: np.ndarray) -> np.ndarray:
        fs = self.params.feature_selection
        thr = self.params.variance_threshold
        if fs is None:
            return X
        if fs == "constant":
            selector = VarianceThreshold(0.0)
        elif fs == "low_variance":
            selector = VarianceThreshold(threshold=thr)
        else:
            raise ValueError(f"Unsupported feature selection: {fs}")
        return selector.fit_transform(X)

    def _reduce_dim(self, X: np.ndarray) -> np.ndarray:
        if self.params.use_pca and X.shape[1] > self.params.pca_components:
            pca = PCA(n_components=self.params.pca_components)
            return pca.fit_transform(X)
        return X

    def _post_transform(self, X: np.ndarray) -> np.ndarray:
        t = self.params.transform_type
        if t is None:
            return X
        if t == "quantile":
            qt = QuantileTransformer(output_distribution="normal")
            return qt.fit_transform(X)
        if t == "power":
            pt = PowerTransformer()
            return pt.fit_transform(X)
        raise ValueError(f"Unsupported transform_type: {t}")
