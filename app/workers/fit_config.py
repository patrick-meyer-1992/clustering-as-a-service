from autocluster import get_evaluator, MetafeatureMapper

from workers.preprocessing import build_preprocess_dict


def prepare_fit_params(df, columns, clustering_algorithms, dim_reduction_algorithms, evaluator_ls, cutoff_time, n_evaluations):
    if not dim_reduction_algorithms:
        dim_reduction_algorithms = ['NullModel']
    if not evaluator_ls:
        evaluator_ls = ['silhouetteScore', 'daviesBouldinScore', 'calinskiHarabaszScore']

    preprocessing_dict = build_preprocess_dict(columns)

    return {
        "df": df,
        "cluster_alg_ls": clustering_algorithms,
        "dim_reduction_alg_ls": dim_reduction_algorithms,
        "optimizer": "smac",
        "n_evaluations": n_evaluations,
        "run_obj": 'quality',
        "seed": 27,
        "cutoff_time": cutoff_time,
        "preprocess_dict": preprocessing_dict,
        "evaluator": get_evaluator(
            evaluator_ls,
            weights=[1, 1, 1],
            clustering_num=None,
            min_proportion=.01
        ),
        "n_folds": 3,
        "warmstart": False,
        "general_metafeatures": MetafeatureMapper.getGeneralMetafeatures(),
        "numeric_metafeatures": MetafeatureMapper.getNumericMetafeatures(),
        "categorical_metafeatures": [],
        "verbose_level": 1,
    }