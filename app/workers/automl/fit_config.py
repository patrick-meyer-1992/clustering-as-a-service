from autocluster import MetafeatureMapper, get_evaluator
from utils.logger import setup_logger

from workers.automl.preprocessing import build_preprocess_dict

logger = setup_logger(__name__)


def prepare_fit_params(
    df, columns, clustering_algorithms, dim_reduction_algorithms, evaluator_ls, cutoff_time, n_evaluations
):
    """
    Prepares the configuration dictionary for fitting the AutoCluster pipeline.

    This function generates all necessary parameters required for the AutoML process,
    including preprocessing settings, evaluator construction, and metafeatures.

    If no dimensionality reduction algorithms or evaluators are provided, defaults are used.

    Parameters:
        df (pandas.DataFrame): The preprocessed dataset to cluster.
        columns (list): List of dictionaries describing the dataset columns.
        clustering_algorithms (list): List of clustering algorithms to evaluate.
        dim_reduction_algorithms (list or None): Dimensionality reduction methods. Defaults to ['NullModel'] if None.
        evaluator_ls (list or None): Evaluation metrics. Defaults to common metrics if None.
        cutoff_time (int): Maximum allowed time (in seconds) per evaluation.
        n_evaluations (int): Total number of evaluation iterations to run.

    Returns:
        dict: A dictionary of keyword arguments suitable for passing to `AutoCluster.fit()`.
    """

    if not dim_reduction_algorithms:
        dim_reduction_algorithms = ["NullModel"]
        logger.warning("No dim_reduction_algorithms provided. Using default: ['NullModel']")

    if not evaluator_ls:
        evaluator_ls = ["silhouetteScore", "daviesBouldinScore", "calinskiHarabaszScore"]
        logger.warning("No evaluator list provided. Using default evaluators.")

    preprocessing_dict = build_preprocess_dict(columns)

    logger.info("Preparing fit parameters for AutoML job")
    logger.debug(f"Selected clustering algorithms: {clustering_algorithms}")
    logger.debug(f"Selected dim reduction algorithms: {dim_reduction_algorithms}")
    logger.debug(f"Selected evaluators: {evaluator_ls}")
    logger.debug(f"Cutoff time: {cutoff_time}, evaluations: {n_evaluations}")
    logger.debug(f"Preprocessing dict: {preprocessing_dict}")

    return {
        "df": df,
        "cluster_alg_ls": clustering_algorithms,
        "dim_reduction_alg_ls": dim_reduction_algorithms,
        "optimizer": "smac",
        "n_evaluations": n_evaluations,
        "run_obj": "quality",
        "seed": 27,
        "cutoff_time": cutoff_time,
        "preprocess_dict": preprocessing_dict,
        "evaluator": get_evaluator(evaluator_ls, weights=[1, 1, 1], clustering_num=None, min_proportion=0.01),
        "n_folds": 3,
        "warmstart": False,
        "general_metafeatures": MetafeatureMapper.getGeneralMetafeatures(),
        "numeric_metafeatures": MetafeatureMapper.getNumericMetafeatures(),
        "categorical_metafeatures": [],
        "verbose_level": 1,
    }
