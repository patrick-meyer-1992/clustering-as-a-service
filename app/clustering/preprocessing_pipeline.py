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
    """
    Reusable preprocessing pipeline for clustering input data.

    This class encapsulates multiple optional preprocessing steps such as imputation,
    outlier removal, feature selection, scaling, normalization, dimensionality reduction,
    and final transformation. Each step is controlled via a `PreProcessingParams` object.

    Typical usage:
    >>> pipeline = PreprocessingPipeline(params)
    >>> processed_X = pipeline.fit_transform(X)

    Parameters
    ----------
    params : PreProcessingParams
        Configuration object defining the preprocessing behavior.
    """

    def __init__(self, params: PreProcessingParams):
        """
        Initialize the preprocessing pipeline with the given parameters.

        Parameters
        ----------
        params : PreProcessingParams
            Configuration object containing options for each preprocessing step.
        """

        self.params = params

    # Optional: keep fitted state for transform-only use
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the full preprocessing pipeline to the input data `X`.

        The steps are applied in the following order:
        1. Imputation (optional)
        2. Outlier removal (optional)
        3. Feature selection (optional)
        4. Scaling
        5. Distribution transform (quantile or power, optional)
        6. Normalization (optional)
        7. Dimensionality reduction via PCA (optional)

        Parameters
        ----------
        X : np.ndarray
            Raw input data.

        Returns
        -------
        np.ndarray
            The processed data after applying all enabled steps.
        """

        X_proc = self._impute(X)
        X_proc = self._remove_outliers(X_proc)
        X_proc = self._feature_select(X_proc)
        X_proc = self._scale(X_proc)
        X_proc = self._post_transform(X_proc)
        X_proc = self._normalize(X_proc)
        X_proc = self._reduce_dim(X_proc)

        return X_proc

    # --- step helpers -----------------------------------------------------
    def _scale(self, X: np.ndarray) -> np.ndarray:
        """
        Apply scaling to the input data using the specified scaler.

        Supported scaler types:
        - "standard": StandardScaler
        - "minmax": MinMaxScaler
        - "robust": RobustScaler
        - "maxabs": MaxAbsScaler
        - "auto": Automatically choose between StandardScaler and RobustScaler
                based on the range of the data.

        Parameters
        ----------
        X : np.ndarray
            The input data to be scaled.

        Returns
        -------
        np.ndarray
            Scaled version of the input data.

        Raises
        ------
        ValueError
            If an unsupported scaler type is specified.
        """

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
        """
        Normalize the input data using the specified normalization type.

        This step ensures each sample has unit norm (L1, L2, or max norm).
        Only applied if `use_normalization` is True in the parameters.

        Parameters
        ----------
        X : np.ndarray
            The input data to normalize.

        Returns
        -------
        np.ndarray
            Normalized data.
        """

        if not self.params.use_normalization:
            return X
        norm = Normalizer(norm=self.params.normalization_type)
        return norm.fit_transform(X)

    def _impute(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values in the input data using the specified strategy.

        Supported strategies:
        - 'mean': Replace missing values with the column mean.
        - 'median': Replace missing values with the column median.
        If no strategy is provided, the data is returned unchanged.

        Parameters
        ----------
        X : np.ndarray
            Input data possibly containing NaNs.

        Returns
        -------
        np.ndarray
            Imputed data with missing values replaced.
        """

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
        """
        Remove outliers from the data based on the configured strategy.

        Supported strategies:
        - 'zscore': Removes rows where any feature exceeds the z-score threshold.
        - 'iqr': Removes rows that lie outside 1.5 IQR of the 25th and 75th percentiles.

        If no strategy is provided, the data is returned unchanged.

        Parameters
        ----------
        X : np.ndarray
            Input data to filter for outliers.

        Returns
        -------
        np.ndarray
            Filtered data with outliers removed.
        """

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
        """
        Select features from the input data based on variance.

        Supported strategies:
        - 'constant': Removes features with zero variance.
        - 'low_variance': Removes features with variance below the configured threshold.

        If no feature selection strategy is set, returns the data unchanged.

        Parameters
        ----------
        X : np.ndarray
            Input data to filter features.

        Returns
        -------
        np.ndarray
            Transformed data with selected features.
        """

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
        """
        Reduce the dimensionality of the input data using PCA.

        PCA is applied only if the `use_pca` flag is enabled and the number of
        features exceeds the target number of components.

        Parameters
        ----------
        X : np.ndarray
            Input data to reduce.

        Returns
        -------
        np.ndarray
            Transformed data with reduced dimensionality.
        """

        if self.params.use_pca and X.shape[1] > self.params.pca_components:
            pca = PCA(n_components=self.params.pca_components)
            return pca.fit_transform(X)
        return X

    def _post_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply an optional post-transformation to the data.

        This step transforms the feature distribution using either quantile transformation
        (for Gaussian-like distribution) or power transformation (for variance stabilization).

        Parameters
        ----------
        X : np.ndarray
            The input data to be transformed.

        Returns
        -------
        np.ndarray
            The transformed data.

        Raises
        ------
        ValueError
            If the specified transform type is not supported.
        """

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
