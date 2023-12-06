"""
Copyright 2022 Pascal Notin, University of Oxford
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np
from typing import AnyStr, List
from slingpy import AbstractDataSource
from slingpy.models.abstract_base_model import AbstractBaseModel
from genedisco.active_learning_methods.acquisition_functions.base_acquisition_function import (
    BaseBatchAcquisitionFunction,
)
from discobax.methods.bax_acquisition import bax_sampling
from discobax.data import abstract_numpy_class

top_k_idx = bax_sampling.top_k_idx


class ThompsonSamplingcquisition(BaseBatchAcquisitionFunction):
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
        if type(dataset_x) == np.ndarray:
            avail_dataset_x = dataset_x[available_indices]
        else:
            avail_dataset_x = dataset_x.subset(available_indices)

        f = (
            model.get_model_prediction(avail_dataset_x, return_multiple_preds=False)[0]
            .flatten()
            .detach()
            .numpy()
            .flatten()
        )

        best_indices = top_k_idx(f, acquisition_batch_size)
        proposal = np.asarray(available_indices)[best_indices]
        return proposal
