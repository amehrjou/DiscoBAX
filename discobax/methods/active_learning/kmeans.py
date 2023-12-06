"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc
"""
import numpy as np
from sklearn.cluster import KMeans
from typing import List, AnyStr
from sklearn.metrics import pairwise_distances
from discobax.methods.active_learning.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)
from slingpy import AbstractDataSource, AbstractBaseModel


class Kmeans(BaseBatchAcquisitionFunction):
    def __init__(self, representation="linear", n_init=10):
        """
        is embedding: Apply kmeans to embedding or raw data
        n_init: Specifies the number of kmeans run-throughs to use, wherein the one with the smallest inertia is selected for the selection phase
        """
        self.representation = representation
        self.n_init = n_init
        super(Kmeans, self).__init__()

    def __call__(
        self,
        dataset_x: AbstractDataSource,
        acquisition_batch_size: int,
        available_indices: List[AnyStr],
        last_selected_indices: List[AnyStr],
        cumulative_indices: List[AnyStr] = None,
        model: AbstractBaseModel = None,
        dataset_y: AbstractDataSource = None,
        temp_folder_name: str = "tmp/model",
    ) -> List:
        if self.representation == "linear":
            kmeans_dataset = model.get_embedding(
                dataset_x.subset(available_indices)
            ).numpy()
        elif self.representation == "raw":
            kmeans_dataset = np.squeeze(dataset_x.subset(available_indices), axis=1)
        else:
            raise ValueError("Representation must be one of 'linear', 'raw'")

        centers = self.kmeans_clustering(kmeans_dataset, acquisition_batch_size)
        chosen = self.select_closest_to_centers(kmeans_dataset, centers)
        return [available_indices[idx] for idx in chosen]

    def kmeans_clustering(self, kmeans_dataset, num_centers):
        kmeans = KMeans(
            init="k-means++", n_init=self.n_init, n_clusters=num_centers
        ).fit(kmeans_dataset)
        return kmeans.cluster_centers_

    def select_closest_to_centers(self, options, centers):
        dist_ctr = pairwise_distances(options, centers)
        min_dist_indices = np.argmin(dist_ctr, axis=0)

        return list(min_dist_indices)
