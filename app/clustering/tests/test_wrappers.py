import numpy as np
import pytest

from app.clustering.wrappers import (
    AffinityPropagationWrapper,
    AgglomerativeClusteringWrapper,
    BayesianGaussianMixtureWrapper,
    BIRCHWrapper,
    BisectingKMeansWrapper,
    DBSCANWrapper,
    GaussianMixtureWrapper,
    HDBSCANWrapper,
    KMeansWrapper,
    MeanShiftWrapper,
    MiniBatchKMeansWrapper,
    OPTICSWrapper,
    SpectralClusteringWrapper,
)

# Dummy test data
dummy_data = np.array(
    [
        [0.0, 0.1],
        [0.2, 0.1],
        [0.1, 0.2],  # Cluster 1
        [10.0, 10.1],
        [10.1, 10.2],
        [10.2, 10.0],  # Cluster 2
        [5.0, 5.1],
        [5.1, 5.0],
        [5.2, 5.3],  # Cluster 3
        [50.0, 50.0],  # Likely to be noise
    ]
)


dummy_columns = {"x": "numeric", "y": "numeric"}

# All wrappers to test (class, params)
clustering_wrappers = [
    (AffinityPropagationWrapper, {}),
    (AgglomerativeClusteringWrapper, {"n_clusters": 2}),
    (BayesianGaussianMixtureWrapper, {}),
    (BIRCHWrapper, {"n_clusters": 3}),
    (BisectingKMeansWrapper, {"n_clusters": 3}),
    (DBSCANWrapper, {}),
    (GaussianMixtureWrapper, {}),
    (HDBSCANWrapper, {}),
    (KMeansWrapper, {"n_clusters": 3}),
    (MeanShiftWrapper, {}),
    (MiniBatchKMeansWrapper, {"n_clusters": 3}),
    (OPTICSWrapper, {}),
    (SpectralClusteringWrapper, {"n_clusters": 3}),
]


@pytest.mark.parametrize("wrapper_cls,params", clustering_wrappers)
def test_wrapper_run_output(wrapper_cls, params):
    wrapper = wrapper_cls(dataset_name="dummy_dataset", columns=dummy_columns, **params)
    result = wrapper.run(dummy_data)

    # Check result type
    assert isinstance(result, dict), f"{wrapper_cls.__name__} should return a dict"

    # Check presence and type of labels
    assert "labels" in result, f"{wrapper_cls.__name__} missing 'labels'"
    assert isinstance(result["labels"], list), f"{wrapper_cls.__name__} labels should be a list"
    assert len(result["labels"]) == dummy_data.shape[0], f"{wrapper_cls.__name__} labels count mismatch"

    # All labels must be integers
    for label in result["labels"]:
        assert isinstance(label, int), f"{wrapper_cls.__name__} label {label} is not an integer"

    # Check presence and type of cluster_sizes
    assert "cluster_sizes" in result, f"{wrapper_cls.__name__} missing 'cluster_sizes'"
    cluster_sizes = result["cluster_sizes"]
    assert isinstance(cluster_sizes, dict), f"{wrapper_cls.__name__} cluster_sizes should be a dict"

    # Remove noise labels (-1, -2, -3) from cluster_sizes
    noise_labels = [-1, -2, -3]
    cluster_sizes_without_noise = {k: v for k, v in cluster_sizes.items() if k not in noise_labels}
    total_labeled = sum(cluster_sizes_without_noise.values())

    # Get number of noise points
    noise_count = result.get("n_noise", result.get("n_noise_", 0))

    # Validate total count
    assert total_labeled + noise_count == dummy_data.shape[0], f"{wrapper_cls.__name__} cluster_sizes count mismatch"

    # If present, n_clusters or n_clusters_ must match number of unique non-noise labels
    for key in ("n_clusters", "n_clusters_"):
        if key in result:
            value = result[key]
            assert isinstance(value, int), f"{wrapper_cls.__name__} '{key}' must be an integer"
            unique_non_noise_labels = set(label for label in result["labels"] if label >= 0)
            assert value == len(unique_non_noise_labels), (
                f"{wrapper_cls.__name__} '{key}' does not match number of unique non-noise labels"
            )

    # Number of unique non-noise labels must match reported n_clusters
    if "n_clusters" in result:
        reported_clusters = result["n_clusters"]
        unique_non_noise_labels = set(label for label in result["labels"] if label >= 0)
        assert reported_clusters == len(unique_non_noise_labels), (
            f"{wrapper_cls.__name__} n_clusters does not match unique non-noise labels"
        )

    # All cluster_sizes keys should be present in labels
    unique_labels = set(result["labels"])
    cluster_keys = set(cluster_sizes.keys())
    assert cluster_keys.issubset(unique_labels), f"{wrapper_cls.__name__} cluster_sizes keys not found in labels"

    # All non-noise labels in labels must be present in cluster_sizes
    for lbl in unique_labels:
        if lbl >= 0:
            assert lbl in cluster_sizes, f"{wrapper_cls.__name__} label {lbl} missing in cluster_sizes"

    # All cluster_sizes values must be positive integers
    for k, v in cluster_sizes.items():
        assert isinstance(v, int) and v > 0, (
            f"{wrapper_cls.__name__} cluster size for label {k} must be positive integer"
        )
