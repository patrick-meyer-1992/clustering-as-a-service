import os
import io
import time
import collections
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from autocluster import AutoCluster, get_evaluator, MetafeatureMapper

from .base_clustering import BaseClustering


class AutoClusterWrapper(BaseClustering):
    backend_name = "automl"
    frontend_name = "AutoML"

    def __init__(self, dataset_name, columns, **params):
        super().__init__(dataset_name, columns)
        self.params = params

        self.default_clustering_algorithms = [
            'KMeans', 'GaussianMixture', 'Birch',
            'MiniBatchKMeans', 'AgglomerativeClustering', 'SpectralClustering'
        ]

        self.default_dim_red_algorithms = [
            'TSNE', 'PCA', 'IncrementalPCA',
            'KernelPCA', 'FastICA', 'TruncatedSVD'
        ]

        self.evaluator_ls = params.get("evaluator_ls", [
            'silhouetteScore', 'daviesBouldinScore', 'calinskiHarabaszScore'
        ])

        self.n_evaluations = params.get("n_evaluations", 50)
        self.cutoff_time = params.get("cutoff_time", 60)
        self.optimizer = params.get("optimizer", "smac")

    @staticmethod
    def get_default_params():
        return {
            "n_evaluations": 50,
            "cutoff_time": 60,
            "optimizer": "smac",
            "evaluator_ls": [
                'silhouetteScore', 'daviesBouldinScore', 'calinskiHarabaszScore'
            ]
        }

    def run(self, data):
        started_timestamp = datetime.now().isoformat()
        try:
            df = pd.DataFrame(data, columns=self.columns)

            cluster = AutoCluster()

            fit_params = {
                "df": df,
                "cluster_alg_ls": self.default_clustering_algorithms,
                "dim_reduction_alg_ls": self.default_dim_red_algorithms,
                "optimizer": self.optimizer,
                "n_evaluations": self.n_evaluations,
                "run_obj": "quality",
                "seed": 27,
                "cutoff_time": self.cutoff_time,
                "preprocess_dict": {
                    "numeric_cols": df.columns.tolist(),
                    "categorical_cols": [],
                    "ordinal_cols": [],
                    "y_col": []
                },
                "evaluator": get_evaluator(
                    self.evaluator_ls,
                    weights=[],
                    clustering_num=None,
                    min_proportion=0.01
                ),
                "n_folds": 3,
                "warmstart": False,
                "general_metafeatures": MetafeatureMapper.getGeneralMetafeatures(),
                "numeric_metafeatures": MetafeatureMapper.getNumericMetafeatures(),
                "categorical_metafeatures": [],
                "verbose_level": 1,
            }

            result_dict = cluster.fit(**fit_params)

            # Optional: Plot erzeugen und speichern
            plot_path = f"plots/{self.dataset_name}_{int(time.time())}.png"
            y_pred = cluster.predict(df, plot=False, save_plot=True, file_path=plot_path)

            result = {
                "labels": y_pred.tolist(),
                "n_clusters": len(set(y_pred)),
                "cluster_sizes": dict(collections.Counter(y_pred)),
                "optimal_cfg": str(result_dict.get("optimal_cfg", "")),
            }

            return result

        except Exception as e:
            print(f"[AutoMLWrapper] Error in run: {e}")
            raise