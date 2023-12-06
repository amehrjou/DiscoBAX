"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc
"""
import numpy as np
from typing import List, AnyStr
from sklearn.metrics import pairwise_distances
from discobax.methods.active_learning.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)
from slingpy import AbstractDataSource, AbstractBaseModel


class CoreSet(BaseBatchAcquisitionFunction):
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
        topmost_hidden_representation = model.get_embedding(
            dataset_x.subset(available_indices)
        ).numpy()
        selected_hidden_representations = model.get_embedding(
            dataset_x.subset(last_selected_indices)
        ).numpy()
        chosen = self.select_most_distant(
            topmost_hidden_representation,
            selected_hidden_representations,
            acquisition_batch_size,
        )
        return [available_indices[idx] for idx in chosen]

    def select_most_distant(self, options, previously_selected, num_samples):
        num_options, num_selected = len(options), len(previously_selected)
        if num_selected == 0:
            min_dist = np.tile(float("inf"), num_options)
        else:
            dist_ctr = pairwise_distances(options, previously_selected)
            min_dist = np.amin(dist_ctr, axis=1)

        indices = []
        for i in range(num_samples):
            idx = min_dist.argmax()
            dist_new_ctr = pairwise_distances(options, options[[idx], :])
            for j in range(num_options):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])
            indices.append(idx)
        return indices
